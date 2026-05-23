"""
Mock Feedback Loop — CPU-only, no diffusers, no real model weights.

Replaces every heavy component with a tiny pure-PyTorch stub so the entire
pipeline can be exercised on any machine in seconds:

    Mocked components             Real components kept
    ─────────────────────         ────────────────────────────────
    StableDiffusionPipeline  →    all training-loop logic
    UNet (4-ch tiny ConvNet) →    gradient flow (real autograd)
    VAE  (4→3 ch upsample)   →    classifier evasion loss
    Scheduler (simplified)   →    config parsing
    Text encoder / tokenizer →    checkpoint & image saving
    Classifier weights       →    JSON history writing
    (random init, no .pth)        all file I/O

Run:
    python feedback_loop_mock.py --config feedback_mock_config.yaml
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_IDX_REAL = 0
CLASSES = 2


# ===========================================================================
# MOCK STABLE DIFFUSION COMPONENTS
# ===========================================================================

class _Attr:
    """Simple namespace helper."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeUNet(nn.Module):
    """Tiny 4-channel conv net that mimics the UNet interface."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.config = _Attr(in_channels=in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, latent_input, t, encoder_hidden_states=None):
        # latent_input: (2B, 4, H, W)  — doubled for classifier-free guidance
        return _Attr(sample=self.net(latent_input))


class FakeVAE(nn.Module):
    """Minimal VAE decoder: 4-ch latent → 3-ch pixel image (×2 upsample)."""

    def __init__(self):
        super().__init__()
        self.config = _Attr(scaling_factor=0.18215)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(4, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # output in [-1, 1], same as real VAE
        )

    def decode(self, latents: torch.Tensor):
        return _Attr(sample=self.net(latents))


class FakeScheduler:
    """Simplified DDIM-like scheduler (Euler step with fixed noise factor)."""

    def __init__(self):
        self.init_noise_sigma: float = 1.0
        self.timesteps: torch.Tensor = torch.linspace(1000, 1, 5).long()
        self.config: dict = {}

    def set_timesteps(self, n: int):
        self.timesteps = torch.linspace(1000, 1, n).long()

    def scale_model_input(self, x: torch.Tensor, t) -> torch.Tensor:
        return x  # no-op in simplified scheduler

    def step(self, noise_pred: torch.Tensor, t, latents: torch.Tensor):
        # Simplified Euler step: move latents slightly away from noise_pred
        prev_sample = latents - 0.05 * noise_pred
        return _Attr(prev_sample=prev_sample)

    @classmethod
    def from_config(cls, config):
        return cls()


class FakeTokenizer:
    """Returns zero token-id tensors of the right shape."""

    def __init__(self):
        self.model_max_length: int = 77

    def __call__(self, texts, padding=None, max_length=None,
                 truncation=None, return_tensors=None):
        ids = torch.zeros(len(texts), self.model_max_length, dtype=torch.long)
        return _Attr(input_ids=ids)


class FakeTextEncoder(nn.Module):
    """Returns zero embeddings — text conditioning has no effect in mock."""

    def forward(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        return (torch.zeros(B, L, 768, device=input_ids.device),)


class FakeSDPipeline:
    """Drop-in replacement for StableDiffusionPipeline."""

    def __init__(self):
        self.unet = FakeUNet(in_channels=4)
        self.vae = FakeVAE()
        self.scheduler = FakeScheduler()
        self.tokenizer = FakeTokenizer()
        self.text_encoder = FakeTextEncoder()

    def to(self, device: torch.device):
        self.unet = self.unet.to(device)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        return self

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        print(f"[MOCK] FakeSDPipeline — skipping download of '{model_id}'")
        return cls()


# ===========================================================================
# MOCK CLASSIFIERS (random weights, no .pth files needed)
# ===========================================================================

def make_mock_classifier(name: str, device: torch.device) -> nn.Module:
    """
    Tiny 2-class CNN with random weights.
    Structurally identical input pipeline to the real classifiers so all
    transform and evasion-loss paths are exercised.
    """
    in_ch = 1 if name.endswith("_dct") else 3
    model = nn.Sequential(
        nn.Conv2d(in_ch, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, CLASSES),
    )
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval().to(device)


# ===========================================================================
# HELPERS  (identical logic to feedback_loop.py)
# ===========================================================================

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


_SPATIAL_STD = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_SPATIAL_VIT = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_classifier_transform(
    pil_img: Image.Image, clf_name: str, log_scale: bool, device: torch.device
) -> torch.Tensor:
    if clf_name.endswith("_dct"):
        return pil_to_dct_tensor(pil_img, log_scale).unsqueeze(0).to(device)
    if clf_name == "vit_b_16":
        return _SPATIAL_VIT(pil_img).unsqueeze(0).to(device)
    return _SPATIAL_STD(pil_img).unsqueeze(0).to(device)


def latent_to_pil(vae: FakeVAE, latents: torch.Tensor) -> list[Image.Image]:
    with torch.no_grad():
        imgs = vae.decode(latents / vae.config.scaling_factor).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    imgs = (imgs * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(arr) for arr in imgs]


# ===========================================================================
# DIFFERENTIABLE GENERATION  (identical to feedback_loop.py)
# ===========================================================================

def generate_latents_differentiable(
    unet: nn.Module,
    scheduler: FakeScheduler,
    text_embeddings: torch.Tensor,
    latent_shape: tuple,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """
    DDIM-style sampling.  All steps except the last run under no_grad.
    The final step keeps the compute graph so loss can propagate into UNet.
    """
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps

    latents = torch.randn(latent_shape, device=device)
    latents = latents * scheduler.init_noise_sigma

    with torch.no_grad():
        for t in timesteps[:-1]:
            latent_input = torch.cat([latents] * 2)
            latent_input = scheduler.scale_model_input(latent_input, t)
            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Final step — WITH gradients
    t = timesteps[-1]
    latent_input = torch.cat([latents] * 2)
    latent_input = scheduler.scale_model_input(latent_input, t)
    noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


# ===========================================================================
# EVASION LOSS  (identical to feedback_loop.py)
# ===========================================================================

def classifier_evasion_loss(
    vae: nn.Module,
    latents: torch.Tensor,
    classifiers: dict,
    log_scale: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    target_label = torch.zeros(latents.shape[0], dtype=torch.long, device=device)

    # Differentiable decode: graph stays alive for spatial classifiers
    pixel_tensors = vae.decode(latents / vae.config.scaling_factor).sample  # B×3×H×W in [-1,1]
    pixel_01 = (pixel_tensors.clamp(-1, 1) + 1) / 2

    total_loss = torch.tensor(0.0, device=device)
    per_clf_info: dict = {}

    for name, clf in classifiers.items():
        if name.endswith("_dct"):
            # DCT path: no differentiable graph through numpy/PIL
            pil_imgs = latent_to_pil(vae, latents.detach())
            tensors = torch.cat(
                [apply_classifier_transform(img, name, log_scale, device) for img in pil_imgs],
                dim=0,
            )
            with torch.no_grad():
                logits = clf(tensors)
            prob_real = torch.softmax(logits.detach(), dim=1)[:, CLASS_IDX_REAL].mean()
            loss_clf = 1.0 - prob_real
            pred_labels = logits.argmax(1)
        else:
            # Spatial path: fully differentiable through VAE pixels
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
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


# ===========================================================================
# PURE-INFERENCE GENERATION (evaluation only — no grad)
# ===========================================================================

def generate_latents_inference(
    unet: nn.Module,
    scheduler: FakeScheduler,
    text_embeddings: torch.Tensor,
    latent_shape: tuple,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Full DDIM pass under no_grad — used by the evaluator, never by the trainer."""
    scheduler.set_timesteps(num_inference_steps)
    latents = torch.randn(latent_shape, device=device) * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, t)
        noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


# ===========================================================================
# EVALUATOR: measure fake-detection rate over N generated images
# ===========================================================================

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
    """
    unet.eval()
    prompts: list[str] = cfg.get("prompts", ["a photo of a person"])
    batch_size: int = int(cfg.get("batch_size", 1))
    num_steps: int  = int(cfg.get("num_inference_steps", 3))
    guidance: float = float(cfg.get("guidance_scale", 7.5))
    latent_h = int(cfg.get("image_height", 64)) // 8
    latent_w = int(cfg.get("image_width",  64)) // 8
    latent_shape = (batch_size, unet.config.in_channels, latent_h, latent_w)

    tag = f" [{label}]" if label else ""
    print(f"\n[MOCK EVAL{tag}] Generating {n_iters} images...")
    counts: dict[str, dict] = {n: {"fake": 0, "real": 0} for n in evaluators}

    with torch.no_grad():
        uncond_ids = tokenizer(
            [""] * batch_size, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        uncond_emb = text_encoder(uncond_ids)[0]

        for i in trange(n_iters, desc=f"Eval{tag}"):
            prompt = prompts[i % len(prompts)]
            cond_ids = tokenizer(
                [prompt] * batch_size, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            text_embeddings = torch.cat([uncond_emb, text_encoder(cond_ids)[0]])

            latents = generate_latents_inference(
                unet, scheduler, text_embeddings,
                latent_shape, num_steps, guidance, device,
            )
            pixel_01 = (vae.decode(latents / vae.config.scaling_factor).sample.clamp(-1, 1) + 1) / 2
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


# ===========================================================================
# MAIN MOCK TRAINING LOOP
# ===========================================================================

def run_feedback_loop_mock(cfg: dict):
    device_str = cfg.get("device", "cpu")
    device = torch.device("cpu")  # mock always runs on CPU
    print(f"[MOCK] Running on CPU (mock mode ignores device setting: '{device_str}')")

    log_scale = bool(cfg.get("dct_log_scale", True))

    # --- Load mock SD pipeline -------------------------------------------
    pipe = FakeSDPipeline.from_pretrained(cfg.get("sd_model_id", "mock"))
    pipe.scheduler = FakeScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    unet = pipe.unet
    vae  = pipe.vae

    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    unet.train()
    unet.requires_grad_(True)
    print(f"[MOCK] FakeUNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # --- Load mock classifiers (training signal) -------------------------
    clf_names: list[str] = cfg.get("classifiers", ["resnet50", "densenet121", "vit_b_16", "densenet121_dct"])
    classifiers: dict[str, nn.Module] = {}
    for name in clf_names:
        classifiers[name] = make_mock_classifier(name, device)
        print(f"[MOCK CLF]  '{name}' (random weights, frozen — training signal)")

    # --- Load mock evaluators (measurement only) -------------------------
    ev_names: list[str] = cfg.get("evaluator", clf_names)
    evaluators: dict[str, nn.Module] = {}
    for name in ev_names:
        evaluators[name] = make_mock_classifier(name, device)
        print(f"[MOCK EVAL] '{name}' (random weights, frozen — measurement only)")

    # --- Settings --------------------------------------------------------
    prompts: list[str]   = cfg.get("prompts", ["a photo of a person"])
    batch_size: int      = int(cfg.get("batch_size", 1))
    num_steps: int       = int(cfg.get("num_inference_steps", 3))
    guidance: float      = float(cfg.get("guidance_scale", 7.5))
    total_iters: int     = int(cfg.get("total_iterations", 10))
    eval_iters: int      = int(cfg.get("eval_iterations", 5))
    save_every: int      = int(cfg.get("save_every", 5))
    save_imgs_every: int = int(cfg.get("save_images_every", 3))
    lr                   = float(cfg.get("learning_rate", 1e-4))
    grad_clip            = float(cfg.get("grad_clip", 1.0))
    output_dir           = Path(cfg.get("output_dir", "mock_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_h = int(cfg.get("image_height", 64)) // 8
    latent_w = int(cfg.get("image_width",  64)) // 8
    latent_shape = (batch_size, unet.config.in_channels, latent_h, latent_w)
    print(f"[MOCK] Latent shape: {latent_shape}  "
          f"(decoded pixels: {batch_size}×3×{latent_h*2}×{latent_w*2})\n")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1 — Pre-training evaluation
    # ═══════════════════════════════════════════════════════════════════════
    pre_results = evaluate_detection(
        unet, vae, pipe.scheduler, pipe.text_encoder, pipe.tokenizer,
        evaluators, cfg, device, eval_iters, log_scale,
        label="PRE-TRAINING",
    )
    print_eval_report(pre_results, "Pre-Training")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2 — Training
    # ═══════════════════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    with torch.no_grad():
        uncond_ids = pipe.tokenizer(
            [""] * batch_size, padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        uncond_emb = pipe.text_encoder(uncond_ids)[0]
        cond_ids = pipe.tokenizer(
            [prompts[0]] * batch_size, padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        cond_emb = pipe.text_encoder(cond_ids)[0]
    text_embeddings = torch.cat([uncond_emb, cond_emb])

    history = []
    print(f"[MOCK] Starting feedback loop — {total_iters} training iterations\n")

    for step in trange(1, total_iters + 1, desc="Mock feedback loop"):
        prompt = prompts[(step - 1) % len(prompts)]
        if step > 1 and (step - 1) % len(prompts) == 0:
            with torch.no_grad():
                cond_ids = pipe.tokenizer(
                    [prompt] * batch_size, padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(device)
                cond_emb = pipe.text_encoder(cond_ids)[0]
            text_embeddings = torch.cat([uncond_emb, cond_emb])

        optimizer.zero_grad()
        latents = generate_latents_differentiable(
            unet=unet, scheduler=pipe.scheduler,
            text_embeddings=text_embeddings, latent_shape=latent_shape,
            num_inference_steps=num_steps, guidance_scale=guidance, device=device,
        )
        loss, clf_info = classifier_evasion_loss(
            vae=vae, latents=latents, classifiers=classifiers,
            log_scale=log_scale, device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
        optimizer.step()

        asr_summary = "  ".join(f"{n}={v['attack_success_rate']:.2f}" for n, v in clf_info.items())
        print(f"\n[Step {step:04d}] loss={loss.item():.4f}  {asr_summary}")
        history.append({"step": step, "loss": loss.item(), "prompt": prompt, "classifiers": clf_info})

        if step % save_imgs_every == 0:
            pil_imgs = latent_to_pil(vae, latents.detach())
            for i, img in enumerate(pil_imgs):
                img.save(output_dir / f"step{step:04d}_img{i}.png")
            print(f"    [MOCK] Saved {len(pil_imgs)} sample image(s) to {output_dir}/")

        if step % save_every == 0:
            ckpt_path = output_dir / f"unet_mock_step{step:04d}.pth"
            torch.save(unet.state_dict(), ckpt_path)
            print(f"    [MOCK] Checkpoint saved: {ckpt_path}")

        with open(output_dir / "mock_training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n[MOCK] Training done.")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3 — Post-training evaluation
    # ═══════════════════════════════════════════════════════════════════════
    post_results = evaluate_detection(
        unet, vae, pipe.scheduler, pipe.text_encoder, pipe.tokenizer,
        evaluators, cfg, device, eval_iters, log_scale,
        label="POST-TRAINING",
    )
    print_eval_report(post_results, "Post-Training")
    print_comparison(pre_results, post_results)

    report = {
        "pre_training":     pre_results,
        "post_training":    post_results,
        "training_history": history,
    }
    with open(output_dir / "mock_training_history.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[MOCK] Full report saved to {output_dir}/mock_training_history.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mock feedback loop — no diffusers, no GPU, no .pth files needed."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_feedback_loop_mock(cfg)
