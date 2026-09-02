"""
Feedback Loop: Fine-tune Stable Diffusion UNet against deepfake classifiers.

Overview
--------
Each iteration of the loop:

1. **Generate** a batch of images from the SD UNet via a full forward diffusion
   + DDIM-inversion-free decode (we keep the latent graph alive with
   `requires_grad=True` on the initial noise so gradients can flow back).
2. **Classify** each generated image with all loaded classifiers.
3. **Compute loss**: we want classifiers to predict "real" (class 0), so we
   minimise cross-entropy against label 0.  The combined loss is the mean
   over all classifiers.
4. **Backpropagate** through the decoder (VAE) and UNet via
   Differentiable Augmentation to update the UNet parameters.
5. **Save** a checkpoint after every `save_every` steps.

Key design decisions
--------------------
* We keep the VAE **frozen** (encoding/decoding is treated as a fixed
  perceptual renderer).  Only the UNet is trained.
* The text encoder is also frozen; only the conditioning embedding is used.
* Gradients through the VAE decode step are enabled so the signal reaches
  the UNet latents.
* A small learning rate (1e-6 default) and gradient clipping prevent
  catastrophic forgetting of the generative quality.
* All classifier weights are frozen throughout.

Usage
-----
    python feedback_loop.py --config feedback_config.yaml

Requirements (in addition to evaluate.py deps)
-----------------------------------------------
    pip install diffusers accelerate transformers
"""

import argparse
import json
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from scipy.fftpack import dct
from tqdm import trange
from torchvision.models import resnet50, densenet121, vit_b_16
from torchvision import models as tv_models

from diffusers import StableDiffusionPipeline, DDIMScheduler

from cavia_utils import LaDeDa9

try:
    import pyiqa
except ImportError:
    pyiqa = None

# ---------------------------------------------------------------------------
# Constants (mirrored from evaluate.py)
# ---------------------------------------------------------------------------
CLASS_IDX_REAL = 0
CLASSES = 2
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Classifier construction helpers (identical to evaluate.py)
# ---------------------------------------------------------------------------

def create_resnet18_dct():
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, CLASSES))
    return model


def create_densenet121_dct():
    model = tv_models.densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(model.classifier.in_features, CLASSES)
    )
    return model


def load_classifier(name: str, weight_path: Path, device: torch.device) -> nn.Module:
    if name == "resnet50":
        model = resnet50()
        model.fc = nn.Linear(model.fc.in_features, CLASSES)
    elif name == "densenet121":
        model = densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, CLASSES)
    elif name == "vit_b_16":
        model = vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, CLASSES)
    elif name == "resnet18_dct":
        model = create_resnet18_dct()
    elif name == "densenet121_dct":
        model = create_densenet121_dct()
    elif name == "cavia2024":
        model = LaDeDa9(num_classes=1)
        model.fc = nn.Linear(2048, 1)
    else:
        raise ValueError(f"Unsupported classifier: {name}")

    # cavia2024 checkpoints wrap the state_dict alongside an argparse.Namespace,
    # which PyTorch 2.6's default weights_only=True refuses to unpickle.
    load_kwargs = {"weights_only": False} if name == "cavia2024" else {}
    state = torch.load(weight_path, map_location=device, **load_kwargs)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    # Freeze all classifier parameters
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


# ---------------------------------------------------------------------------
# No-reference IQA metric loader (differentiable perceptual-quality term)
# ---------------------------------------------------------------------------

def load_iqa_metric(name: str, device: torch.device):
    """Load a pyiqa NR-IQA metric wrapped as a differentiable loss.

    Returns a callable f(x) → scalar where x is B×3×H×W in [0,1] and the
    returned value is a LOSS (lower = better). Higher-is-better metrics are
    inverted to (1 - score) so the caller can always minimise.
    """
    if pyiqa is None:
        raise ImportError("pyiqa is required for the IQA loss; run `pip install pyiqa`.")
    metric = pyiqa.create_metric(name, as_loss=True, device=device)
    for p in metric.parameters():
        p.requires_grad_(False)
    metric.eval()
    # pyiqa exposes `lower_better` on the metric object; default assumption is
    # higher-is-better if the attribute is missing.
    higher_better = not bool(getattr(metric, "lower_better", False))

    def _loss(x: torch.Tensor) -> torch.Tensor:
        score = metric(x)
        return (1.0 - score).mean() if higher_better else score.mean()

    return _loss


# ---------------------------------------------------------------------------
# Transforms for classifiers (mirrored from evaluate.py)
# ---------------------------------------------------------------------------

def dct2(np_img: np.ndarray) -> np.ndarray:
    return dct(dct(np_img, axis=0, norm="ortho"), axis=1, norm="ortho")


def pil_to_dct_tensor(pil_img: Image.Image, log_scale: bool) -> torch.Tensor:
    img = pil_img.convert("L")
    if max(img.size) > 256:
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
    w, h = img.size
    left, top = (w - 128) // 2, (h - 128) // 2
    img = img.crop((left, top, left + 128, top + 128))
    np_img = np.array(img, dtype=np.float32)
    dct_img = dct2(np_img)
    if log_scale:
        dct_img = np.log(np.abs(dct_img) + 1e-6)
    return torch.from_numpy(dct_img).unsqueeze(0)  # 1×128×128


SPATIAL_TRANSFORM_STANDARD = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SPATIAL_TRANSFORM_VIT = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_classifier_transform(
    pil_img: Image.Image,
    clf_name: str,
    log_scale: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return a (1, C, H, W) tensor ready for the given classifier."""
    if clf_name.endswith("_dct"):
        return pil_to_dct_tensor(pil_img, log_scale).unsqueeze(0).to(device)
    if clf_name == "vit_b_16":
        return SPATIAL_TRANSFORM_VIT(pil_img).unsqueeze(0).to(device)
    return SPATIAL_TRANSFORM_STANDARD(pil_img).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Differentiable image conversion: latent tensor → PIL
# ---------------------------------------------------------------------------

def latent_to_pil(vae, latents: torch.Tensor) -> list[Image.Image]:
    """Decode VAE latents to a list of PIL images (no grad)."""
    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        imgs = vae.decode((latents / vae.config.scaling_factor).to(vae_dtype)).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # [0, 1]
    imgs = (imgs * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(arr) for arr in imgs]


# ---------------------------------------------------------------------------
# Differentiable generation step
# ---------------------------------------------------------------------------

def generate_latents_differentiable(
    unet,
    scheduler,
    text_embeddings: torch.Tensor,
    latent_shape: tuple,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    generator: torch.Generator | None = None,
    ref_unet: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Run DDIM sampling keeping gradients alive through the final UNet step.

    All denoising steps except the last are run under `torch.no_grad()` for
    memory efficiency.  The final step is run with gradients so that the loss
    can propagate back through UNet → latent → VAE decode.

    If ``ref_unet`` is provided, the frozen reference is run on the same
    latent/timestep and an MSE distillation loss on the raw noise prediction
    is returned alongside the latents.  This is the anti-drift signal that
    keeps the trainable UNet close to the original SD prior.
    """
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    unet_dtype = next(unet.parameters()).dtype
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=unet_dtype)
    latents = latents * scheduler.init_noise_sigma
    text_embeddings = text_embeddings.to(dtype=unet_dtype)

    # All steps except the last: no grad (saves memory)
    with torch.no_grad():
        for t in timesteps[:-1]:
            latent_input = torch.cat([latents] * 2)  # for CFG
            latent_input = scheduler.scale_model_input(latent_input, t)
            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Final step WITH gradients
    t = timesteps[-1]
    latent_input = torch.cat([latents] * 2)
    latent_input = scheduler.scale_model_input(latent_input, t)
    noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample

    distill_loss: torch.Tensor | None = None
    if ref_unet is not None:
        with torch.no_grad():
            ref_noise_pred = ref_unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        distill_loss = F.mse_loss(noise_pred, ref_noise_pred.detach())

    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents, distill_loss


# ---------------------------------------------------------------------------
# Loss: classifiers must predict "real" for generated images
# ---------------------------------------------------------------------------

def classifier_evasion_loss(
    vae,
    latents: torch.Tensor,
    classifiers: dict,
    log_scale: bool,
    device: torch.device,
    iqa_metric=None,
    iqa_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Decode latents → PIL images, run every classifier, return combined loss.

    We attach gradients from the pixel domain back to `latents` via the VAE
    decode.  Because PIL conversions break the graph, we use a surrogate:
    we compute the classifier logits on PIL images (no grad for the classifier
    forward pass itself) and construct a soft loss that is differentiable
    with respect to the VAE output tensor.

    Strategy
    --------
    1. Decode latents WITH grad tracking to get pixel tensors.
    2. For each classifier, resize/normalise the pixel tensor directly
       (no PIL round-trip) so the graph is intact.
    3. Run frozen classifier → logits → CE loss vs. label 0 ("real").
    """
    target_label = torch.zeros(latents.shape[0], dtype=torch.long, device=device)  # class 0 = real

    # Decode with grad
    vae_dtype = next(vae.parameters()).dtype
    pixel_tensors = vae.decode((latents / vae.config.scaling_factor).to(vae_dtype)).sample  # B×3×H×W in [-1,1]
    pixel_01 = (pixel_tensors.clamp(-1, 1) + 1) / 2  # [0, 1]

    total_loss = torch.tensor(0.0, device=device)
    per_clf_info = {}

    for name, clf in classifiers.items():
        if name.endswith("_dct"):
            # DCT classifiers: convert to PIL then to numpy (no grad path)
            # We detach here because the DCT is not differentiable in PyTorch;
            # the spatial classifiers carry the gradient signal.
            pil_imgs = latent_to_pil(vae, latents.detach())
            tensors = torch.cat(
                [apply_classifier_transform(img, name, log_scale, device) for img in pil_imgs],
                dim=0,
            )
            with torch.no_grad():
                logits = clf(tensors)
            # Surrogate: use softmax prob of "real" as a differentiable proxy
            # attached to the spatial path (pixel_01 mean as anchor)
            prob_real = torch.softmax(logits.detach(), dim=1)[:, CLASS_IDX_REAL].mean()
            # Loss: we want prob_real → 1, so minimise (1 - prob_real)
            # This is a constant w.r.t. UNet params but provides a meaningful metric
            loss_clf = (1.0 - prob_real)
            pred_labels = logits.argmax(1)
        else:
            # Spatial classifiers: differentiable path through VAE pixels
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            if name == "vit_b_16":
                resized = F.interpolate(pixel_01, size=(224, 224), mode="bilinear", align_corners=False)
            else:
                resized = F.interpolate(pixel_01, size=(256, 256), mode="bilinear", align_corners=False)
            normalised = (resized - mean) / std
            logits = clf(normalised)
            if name == "cavia2024":
                # LaDeDa9 emits a single logit per image; convert to 2-class
                # form matching CLASS_IDX_REAL=0 (higher raw logit = more "fake").
                logits = torch.cat([-logits, logits], dim=-1)
            loss_clf = F.cross_entropy(logits, target_label)
            pred_labels = logits.argmax(1)

        attack_success = (pred_labels == CLASS_IDX_REAL).float().mean().item()
        total_loss = total_loss + loss_clf
        per_clf_info[name] = {
            "loss": loss_clf.item(),
            "attack_success_rate": attack_success,
        }

    detector_loss = total_loss / len(classifiers)
    if iqa_metric is not None and iqa_weight > 0:
        iqa_loss = iqa_metric(pixel_01.float())
        per_clf_info["_iqa"] = {"loss": iqa_loss.item(), "weight": iqa_weight}
        return detector_loss + iqa_weight * iqa_loss, per_clf_info
    return detector_loss, per_clf_info


# ---------------------------------------------------------------------------
# Pure-inference generation (evaluation only — no grad)
# ---------------------------------------------------------------------------

def generate_latents_inference(
    unet,
    scheduler,
    text_embeddings: torch.Tensor,
    latent_shape: tuple,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Full DDIM pass under no_grad — used by the evaluator, never by the trainer."""
    scheduler.set_timesteps(num_inference_steps)
    unet_dtype = next(unet.parameters()).dtype
    latents = torch.randn(latent_shape, device=device, dtype=unet_dtype) * scheduler.init_noise_sigma
    text_embeddings = text_embeddings.to(dtype=unet_dtype)
    for t in scheduler.timesteps:
        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, t)
        noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


# ---------------------------------------------------------------------------
# Evaluator: measure fake-detection rate over N generated images
# ---------------------------------------------------------------------------

def evaluate_detection(
    unet: nn.Module,
    vae: nn.Module,
    scheduler,
    text_encoder: nn.Module,
    tokenizer,
    evaluators: dict,
    cfg: dict,
    device: torch.device,
    n_iters: int,
    log_scale: bool,
    label: str = "",
) -> dict:
    """
    Generate n_iters images with the current UNet and count how many each
    evaluator classifies as fake (class 1) vs. real (class 0).
    Returns per-evaluator stats and an aggregate.
    """
    unet.eval()
    prompt: str = cfg.get("prompt", "a photo of a person")
    negative_prompt: str = cfg.get("negative_prompt", "")
    batch_size: int = int(cfg.get("batch_size", 1))
    num_steps: int = int(cfg.get("num_inference_steps", 20))
    guidance: float = float(cfg.get("guidance_scale", 7.5))
    latent_h = int(cfg.get("image_height", 512)) // 8
    latent_w = int(cfg.get("image_width", 512)) // 8
    latent_shape = (batch_size, unet.config.in_channels, latent_h, latent_w)

    tag = f" [{label}]" if label else ""
    print(f"\n[EVAL{tag}] Generating {n_iters} images...")
    counts: dict[str, dict] = {n: {"fake": 0, "real": 0} for n in evaluators}

    with torch.no_grad():
        uncond_ids = tokenizer(
            [negative_prompt] * batch_size, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        uncond_emb = text_encoder(uncond_ids)[0]
        cond_ids = tokenizer(
            [prompt] * batch_size, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        text_embeddings = torch.cat([uncond_emb, text_encoder(cond_ids)[0]])

        for i in trange(n_iters, desc=f"Eval{tag}"):
            latents = generate_latents_inference(
                unet, scheduler, text_embeddings,
                latent_shape, num_steps, guidance, device,
            )
            vae_dtype = next(vae.parameters()).dtype
            pixel_01 = (vae.decode((latents / vae.config.scaling_factor).to(vae_dtype)).sample.clamp(-1, 1) + 1) / 2
            pil_imgs = latent_to_pil(vae, latents)

            for name, clf in evaluators.items():
                if name.endswith("_dct"):
                    tensors = torch.cat(
                        [apply_classifier_transform(img, name, log_scale, device) for img in pil_imgs],
                        dim=0,
                    )
                    logits = clf(tensors)
                else:
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    size = (224, 224) if name == "vit_b_16" else (256, 256)
                    logits = clf((F.interpolate(pixel_01, size=size, mode="bilinear", align_corners=False) - mean) / std)
                    if name == "cavia2024":
                        logits = torch.cat([-logits, logits], dim=-1)
                for pred in logits.argmax(1):
                    counts[name]["real" if pred.item() == CLASS_IDX_REAL else "fake"] += 1

    unet.train()

    results: dict = {}
    for name, c in counts.items():
        total = c["fake"] + c["real"]
        results[name] = {
            "detected_fake": c["fake"],
            "detected_real": c["real"],
            "total": total,
            "fake_detection_rate": c["fake"] / total if total > 0 else 0.0,
            "evasion_rate":        c["real"] / total if total > 0 else 0.0,
        }
    agg_fake  = sum(v["detected_fake"] for v in results.values())
    agg_total = sum(v["total"]         for v in results.values())
    results["_aggregate"] = {
        "fake_detection_rate": agg_fake / agg_total if agg_total > 0 else 0.0,
        "evasion_rate":        1.0 - (agg_fake / agg_total) if agg_total > 0 else 0.0,
    }
    return results


def print_eval_report(results: dict, label: str):
    W = 64
    print(f"\n{'═'*W}")
    print(f"  Evaluation Report — {label}")
    print(f"{'═'*W}")
    for name, s in results.items():
        if name == "_aggregate":
            continue
        filled = round(s["fake_detection_rate"] * 20)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"  {name:<17}  detected fake: {s['detected_fake']:>4}/{s['total']:<4}"
              f"  [{bar}] {s['fake_detection_rate']:>6.1%}")
    agg = results["_aggregate"]
    print(f"  {'─'*60}")
    filled = round(agg["fake_detection_rate"] * 20)
    bar = "█" * filled + "░" * (20 - filled)
    print(f"  {'AGGREGATE':<17}  detection rate:        [{bar}] {agg['fake_detection_rate']:>6.1%}")
    print(f"  {'':17}  evasion  rate:         {'':22}{agg['evasion_rate']:>6.1%}")
    print(f"{'═'*W}\n")


def print_comparison(pre: dict, post: dict):
    W = 64
    pre_agg  = pre["_aggregate"]["fake_detection_rate"]
    post_agg = post["_aggregate"]["fake_detection_rate"]
    delta    = post_agg - pre_agg
    arrow    = "▼" if delta < 0 else ("▲" if delta > 0 else "─")
    print(f"\n{'═'*W}")
    print(f"  Training Effect Summary")
    print(f"{'═'*W}")
    print(f"  {'Evaluator':<17}  {'Before':>8}   {'After':>8}   {'Δ':>7}")
    print(f"  {'─'*60}")
    for name in (n for n in pre if n != "_aggregate"):
        p = pre[name]["fake_detection_rate"]
        q = post[name]["fake_detection_rate"]
        d = q - p
        sym = "▼" if d < 0 else ("▲" if d > 0 else "─")
        print(f"  {name:<17}  {p:>8.1%}   {q:>8.1%}   {sym}{abs(d):>5.1%}")
    print(f"  {'─'*60}")
    print(f"  {'AGGREGATE':<17}  {pre_agg:>8.1%}   {post_agg:>8.1%}   {arrow}{abs(delta):>5.1%}")
    direction = "decreased" if delta < 0 else "increased"
    print(f"\n  Fake detection {direction} by {abs(delta):.1%} after training.")
    print(f"  Evasion rate: {pre['_aggregate']['evasion_rate']:.1%} "
          f"→ {post['_aggregate']['evasion_rate']:.1%}")
    print(f"{'═'*W}\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_feedback_loop(cfg: dict):
    device_str = cfg.get("device", "auto")
    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    print(f"[DEVICE] {device}")

    # --- Load SD pipeline ---------------------------------------------------
    sd_model_id = cfg["sd_model_id"]
    print(f"[SD] Loading Stable Diffusion from '{sd_model_id}'...")
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    unet         = pipe.unet
    vae          = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer    = pipe.tokenizer

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.float()  # upcast to fp32 for stable training; fp16 AdamW causes gradient overflow → NaN → black images
    unet.train()
    unet.requires_grad_(True)
    print(f"[SD] UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

    models_dir = Path(cfg["models_dir"])
    log_scale  = bool(cfg.get("dct_log_scale", True))

    # --- Load classifiers (adversarial training signal) --------------------
    classifiers: dict[str, nn.Module] = {}
    for clf_name in cfg["classifiers"]:
        w_path = models_dir / f"{clf_name}.pth"
        if not w_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {w_path}")
        classifiers[clf_name] = load_classifier(clf_name, w_path, device)
        print(f"[CLF]  Loaded '{clf_name}' (frozen — training signal)")

    # --- Load evaluators (measurement only, no gradient) ------------------
    evaluator_names: list[str] = cfg.get("evaluator", cfg["classifiers"])
    evaluators: dict[str, nn.Module] = {}
    for ev_name in evaluator_names:
        w_path = models_dir / f"{ev_name}.pth"
        if not w_path.exists():
            raise FileNotFoundError(f"Evaluator weights not found: {w_path}")
        evaluators[ev_name] = load_classifier(ev_name, w_path, device)
        print(f"[EVAL] Loaded '{ev_name}' (frozen — measurement only)")

    # --- Settings -----------------------------------------------------------
    prompt: str          = cfg.get("prompt", "a photo of a person")
    negative_prompt: str = cfg.get("negative_prompt", "")
    batch_size: int      = int(cfg.get("batch_size", 1))
    num_steps: int       = int(cfg.get("num_inference_steps", 20))
    guidance: float      = float(cfg.get("guidance_scale", 7.5))
    total_iterations     = int(cfg.get("total_iterations", 500))
    eval_iterations      = int(cfg.get("eval_iterations", 100))
    save_every           = int(cfg.get("save_every", 50))
    save_images_every    = int(cfg.get("save_images_every", 25))
    lr                   = float(cfg.get("learning_rate", 1e-6))
    grad_clip            = float(cfg.get("grad_clip", 1.0))
    output_dir           = Path(cfg.get("output_dir", "feedback_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional IQA loss --------------------------------------------------
    iqa_name   = str(cfg.get("iqa_metric", "clipiqa"))
    iqa_weight = float(cfg.get("iqa_weight", 0.0))
    iqa_metric = None
    if iqa_weight > 0:
        iqa_metric = load_iqa_metric(iqa_name, device)
        print(f"[IQA]  Loaded '{iqa_name}' (weight={iqa_weight}, frozen)")

    # --- Optional anti-drift distillation to a frozen reference UNet -------
    # Deepcopy the current UNet BEFORE any training so the reference captures
    # the original SD prior. Only the trainable UNet's noise prediction gets
    # gradients; the reference is frozen and eval-mode.
    distill_weight = float(cfg.get("distill_weight", 0.0))
    ref_unet: nn.Module | None = None
    if distill_weight > 0:
        ref_unet = deepcopy(unet)
        ref_unet.eval()
        ref_unet.requires_grad_(False)
        print(f"[DISTILL] Loaded frozen reference UNet (weight={distill_weight})")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1 — Pre-training evaluation
    # ═══════════════════════════════════════════════════════════════════════
    pre_results = evaluate_detection(
        unet, vae, pipe.scheduler, text_encoder, tokenizer,
        evaluators, cfg, device, eval_iterations, log_scale,
        label="PRE-TRAINING",
    )
    print_eval_report(pre_results, "Pre-Training")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2 — Training
    # ═══════════════════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    with torch.no_grad():
        uncond_ids = tokenizer(
            [negative_prompt] * batch_size, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        uncond_emb = text_encoder(uncond_ids)[0]
        cond_ids = tokenizer(
            [prompt] * batch_size, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        cond_emb = text_encoder(cond_ids)[0]
    text_embeddings = torch.cat([uncond_emb, cond_emb])

    latent_h     = int(cfg.get("image_height", 512)) // 8
    latent_w     = int(cfg.get("image_width",  512)) // 8
    latent_shape = (batch_size, unet.config.in_channels, latent_h, latent_w)

    history = []
    print(f"\n[LOOP] Starting feedback loop for {total_iterations} iterations\n")

    for step in trange(1, total_iterations + 1, desc="Feedback loop"):
        optimizer.zero_grad()

        latents, distill_loss = generate_latents_differentiable(
            unet=unet, scheduler=pipe.scheduler,
            text_embeddings=text_embeddings, latent_shape=latent_shape,
            num_inference_steps=num_steps, guidance_scale=guidance, device=device,
            ref_unet=ref_unet,
        )
        loss, clf_info = classifier_evasion_loss(
            vae=vae, latents=latents, classifiers=classifiers,
            log_scale=log_scale, device=device,
            iqa_metric=iqa_metric, iqa_weight=iqa_weight,
        )
        if distill_loss is not None:
            loss = loss + distill_weight * distill_loss
            clf_info["_distill"] = {"loss": distill_loss.item(), "weight": distill_weight}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
        optimizer.step()

        history.append({"step": step, "loss": loss.item(), "prompt": prompt, "classifiers": clf_info})
        asr_summary = "  ".join(
            f"{n}={v['attack_success_rate']:.2f}"
            for n, v in clf_info.items() if not n.startswith("_")
        )
        iqa_str = f"  iqa={clf_info['_iqa']['loss']:.3f}" if "_iqa" in clf_info else ""
        distill_str = f"  distill={clf_info['_distill']['loss']:.4f}" if "_distill" in clf_info else ""
        print(f"\n[Step {step:04d}] loss={loss.item():.4f}  {asr_summary}{iqa_str}{distill_str}")

        if step % save_images_every == 0:
            pil_imgs = latent_to_pil(vae, latents.detach())
            for i, img in enumerate(pil_imgs):
                img.save(output_dir / f"step{step:04d}_img{i}.png")
            print(f"    Saved {len(pil_imgs)} sample image(s) to {output_dir}/")

        if step % save_every == 0:
            ckpt_path = output_dir / f"unet_step{step:04d}.pth"
            torch.save(unet.state_dict(), ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    final_ckpt = output_dir / "unet_final.pth"
    torch.save(unet.state_dict(), final_ckpt)
    print(f"\n[DONE] Final UNet weights: {final_ckpt}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3 — Post-training evaluation
    # ═══════════════════════════════════════════════════════════════════════
    post_results = evaluate_detection(
        unet, vae, pipe.scheduler, text_encoder, tokenizer,
        evaluators, cfg, device, eval_iterations, log_scale,
        label="POST-TRAINING",
    )
    print_eval_report(post_results, "Post-Training")
    print_comparison(pre_results, post_results)

    report = {
        "pre_training":      pre_results,
        "post_training":     post_results,
        "training_history":  history,
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[DONE] Full report saved to {output_dir}/training_history.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD feedback loop against deepfake classifiers.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_feedback_loop(cfg)
