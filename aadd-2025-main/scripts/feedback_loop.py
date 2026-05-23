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
    model = tv_models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, CLASSES))
    return model


def create_densenet121_dct():
    model = tv_models.densenet121(pretrained=False)
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
    else:
        raise ValueError(f"Unsupported classifier: {name}")

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    # Freeze all classifier parameters
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


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
    with torch.no_grad():
        imgs = vae.decode(latents / vae.config.scaling_factor).sample
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
) -> torch.Tensor:
    """
    Run DDIM sampling keeping gradients alive through the final UNet step.

    All denoising steps except the last are run under `torch.no_grad()` for
    memory efficiency.  The final step is run with gradients so that the loss
    can propagate back through UNet → latent → VAE decode.
    """
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    latents = torch.randn(latent_shape, generator=generator, device=device)
    latents = latents * scheduler.init_noise_sigma

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
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents  # gradients flow through this


# ---------------------------------------------------------------------------
# Loss: classifiers must predict "real" for generated images
# ---------------------------------------------------------------------------

def classifier_evasion_loss(
    vae,
    latents: torch.Tensor,
    classifiers: dict,
    log_scale: bool,
    device: torch.device,
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
    pixel_tensors = vae.decode(latents / vae.config.scaling_factor).sample  # B×3×H×W in [-1,1]
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
            loss_clf = F.cross_entropy(logits, target_label)
            pred_labels = logits.argmax(1)

        attack_success = (pred_labels == CLASS_IDX_REAL).float().mean().item()
        total_loss = total_loss + loss_clf
        per_clf_info[name] = {
            "loss": loss_clf.item(),
            "attack_success_rate": attack_success,
        }

    return total_loss / len(classifiers), per_clf_info


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
        safety_checker=None,  # disabled — we are doing research
    )
    # Replace scheduler with DDIM for deterministic stepping
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # UNet is trainable
    unet.train()
    unet.requires_grad_(True)

    print(f"[SD] UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # --- Load classifiers ---------------------------------------------------
    models_dir = Path(cfg["models_dir"])
    log_scale = bool(cfg.get("dct_log_scale", True))
    classifiers: dict[str, nn.Module] = {}
    for clf_name in cfg["classifiers"]:
        w_path = models_dir / f"{clf_name}.pth"
        if not w_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {w_path}")
        classifiers[clf_name] = load_classifier(clf_name, w_path, device)
        print(f"[CLF] Loaded '{clf_name}' (frozen)")

    # --- Optimizer ----------------------------------------------------------
    lr = float(cfg.get("learning_rate", 1e-6))
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    grad_clip = float(cfg.get("grad_clip", 1.0))

    # --- Generation settings ------------------------------------------------
    prompts: list[str] = cfg.get("prompts", ["a photo of a person"])
    batch_size: int = int(cfg.get("batch_size", 1))
    num_steps: int = int(cfg.get("num_inference_steps", 20))
    guidance: float = float(cfg.get("guidance_scale", 7.5))
    total_iterations: int = int(cfg.get("total_iterations", 500))
    save_every: int = int(cfg.get("save_every", 50))
    output_dir = Path(cfg.get("output_dir", "feedback_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_images_every: int = int(cfg.get("save_images_every", 25))

    # Pre-compute text embeddings (CFG: unconditional + conditional)
    with torch.no_grad():
        # Unconditional embedding
        uncond_ids = tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        uncond_emb = text_encoder(uncond_ids)[0]

        # Conditional embeddings (cycle through prompts)
        cond_ids = tokenizer(
            [prompts[0]] * batch_size,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        cond_emb = text_encoder(cond_ids)[0]

    text_embeddings = torch.cat([uncond_emb, cond_emb])  # 2B × 77 × 768

    # Latent shape: SD uses 4-channel latents at 1/8 spatial resolution
    latent_h = int(cfg.get("image_height", 512)) // 8
    latent_w = int(cfg.get("image_width", 512)) // 8
    latent_shape = (batch_size, unet.config.in_channels, latent_h, latent_w)

    # --- Training -----------------------------------------------------------
    history = []
    print(f"\n[LOOP] Starting feedback loop for {total_iterations} iterations\n")

    for step in trange(1, total_iterations + 1, desc="Feedback loop"):
        # Rotate prompt every iteration
        prompt = prompts[(step - 1) % len(prompts)]
        if step > 1 and (step - 1) % len(prompts) == 0:
            # Refresh conditional embedding when prompt rotates
            with torch.no_grad():
                cond_ids = tokenizer(
                    [prompt] * batch_size,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
                cond_emb = text_encoder(cond_ids)[0]
            text_embeddings = torch.cat([uncond_emb, cond_emb])

        optimizer.zero_grad()

        # 1. Generate latents (differentiable through final UNet step)
        latents = generate_latents_differentiable(
            unet=unet,
            scheduler=pipe.scheduler,
            text_embeddings=text_embeddings,
            latent_shape=latent_shape,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            device=device,
        )

        # 2. Compute evasion loss
        loss, clf_info = classifier_evasion_loss(
            vae=vae,
            latents=latents,
            classifiers=classifiers,
            log_scale=log_scale,
            device=device,
        )

        # 3. Backprop + update UNet
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
        optimizer.step()

        # 4. Logging
        log_entry = {
            "step": step,
            "loss": loss.item(),
            "prompt": prompt,
            "classifiers": clf_info,
        }
        history.append(log_entry)

        asr_summary = "  ".join(
            f"{n}={v['attack_success_rate']:.2f}" for n, v in clf_info.items()
        )
        trange_msg = f"loss={loss.item():.4f}  {asr_summary}"
        print(f"\n[Step {step:04d}] {trange_msg}")

        # 5. Save sample images periodically
        if step % save_images_every == 0:
            pil_imgs = latent_to_pil(vae, latents.detach())
            for i, img in enumerate(pil_imgs):
                img.save(output_dir / f"step{step:04d}_img{i}.png")
            print(f"    Saved {len(pil_imgs)} sample image(s) to {output_dir}/")

        # 6. Save UNet checkpoint periodically
        if step % save_every == 0:
            ckpt_path = output_dir / f"unet_step{step:04d}.pth"
            torch.save(unet.state_dict(), ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")

        # 7. Save JSON history
        json_path = output_dir / "training_history.json"
        with open(json_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n[DONE] Final checkpoint and history saved to {output_dir}/")
    final_ckpt = output_dir / "unet_final.pth"
    torch.save(unet.state_dict(), final_ckpt)
    print(f"[DONE] Final UNet weights: {final_ckpt}")


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
