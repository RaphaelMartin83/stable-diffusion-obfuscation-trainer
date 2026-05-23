# AADD-2025: Adversarial Attacks on Deepfake Detectors Challenge

<picture>
  <img alt="Logos" src="/assets/images/headernewaadd.jpg">
</picture>

## 1st Adversarial Attacks on Deepfake Detectors: A Challenge in the Era of AI-Generated Media

**Grand Challenge at [ACM Multimedia 2025](https://acmmm2025.org/)**

---

## 🎯 Overview

The AADD-2025 Challenge investigated adversarial vulnerabilities of deepfake detection models by generating adversarial perturbed deepfake images that evade standard classifiers while maintaining high visual similarity to the original deepfake content. Given the increasing reliance on deepfake detectors in forensic analysis and content moderation, ensuring their robustness against adversarial attacks has relevant importance.

## 🎪 Challenge Description

The goal of this challenge was to expose and address vulnerabilities in current deepfake detection systems by designing adversarial attacks that alter deepfake images, rendering them unrecognizable as synthetic content to 4 proposed classifiers, **preserving high visual similarity** to the original images.

## 📊 Dataset Structure

Participants were provided with a dataset divided into **sixteen subsets**:

### High Quality Resolution:
- **4 GAN-based models** (high quality)
- **4 Diffusion-based models** (high quality)

### Low Quality Resolution:
- **4 GAN-based models** (low quality)
- **4 Diffusion-based models** (low quality)

```
- Dataset
├── train
│   ├── fake
│   │   ├── hq
│   │   │   ├── Adobe Firefly
│   │   │   ├── Deep AI
│   │   │   ├── Flux.1.1 Pro
│   │   │   ├── Hotpot AI
│   │   │   ├── Nvidia Sana PAG
│   │   │   ├── Stable Diffusion 3.5
│   │   │   ├── StyleGAN2
│   │   │   ├── StyleGAN3
│   │   │   └── Tencent Hunyuan
│   │   └── lq
│   │       ├── Deep AI
│   │       ├── Flux.1
│   │       ├── Freepik
│   │       ├── Hotpot AI
│   │       ├── Nvidia Sana PAG
│   │       ├── Stable Diffusion Attend and Excite
│   │       ├── StyleGAN
│   │       ├── StyleGAN3
│   │       └── Tencent Hunyuan
│   └── real
│       ├── hq
│       │   └── ffhq
│       └── lq
│           └── celeba_hq
└── test
    ├── fake
    │   ├── hq
    │   │   ├── Adobe Firefly
    │   │   ├── Deep AI
    │   │   ├── Flux.1.1 Pro
    │   │   ├── Hotpot AI
    │   │   ├── Nvidia Sana PAG
    │   │   ├── Stable Diffusion 3.5
    │   │   ├── StyleGAN2
    │   │   ├── StyleGAN3
    │   │   └── Tencent Hunyuan
    │   └── lq
    │       ├── Deep AI
    │       ├── Flux.1
    │       ├── Freepik
    │       ├── Hotpot AI
    │       ├── Nvidia Sana PAG
    │       ├── Stable Diffusion Attend and Excite
    │       ├── StyleGAN
    │       ├── StyleGAN3
    │       └── Tencent Hunyuan
    └── real
        ├── hq
        │   └── ffhq
        └── lq
            └── celeba_hq
```

**Note**: Participants had to focus on the entire dataset across all subsets.

## 📋 Submission Requirements

1. **Adversarial Images**: Submit the generated adversarial deepfake images
2. **Technical Abstract**: Provide a detailed description of your methodology
3. **Results Documentation**: Include performance metrics and analysis

## 📥 Evaluation Resources

**Final Evaluation Scripts** [See here](scripts)

---

## 🚀 Getting Started

This repository includes:
- **`scripts/evaluate.py`** — evaluates adversarial robustness of classifiers against an existing adversarial image set
- **`scripts/feedback_loop.py`** — fine-tunes a Stable Diffusion UNet so its output evades the classifiers (requires CUDA)
- **`scripts/feedback_loop_mock.py`** — identical pipeline with tiny stub models for CPU testing (no GPU, no model download)

### Repository layout

```
aadd-2025-main/
├── code/
│   └── classifier.py
├── scripts/
│   ├── config.yaml                 ← evaluate.py config
│   ├── evaluate.py
│   ├── feedback_config.yaml        ← real feedback loop config
│   ├── feedback_loop.py
│   ├── feedback_mock_config.yaml   ← mock feedback loop config
│   └── feedback_loop_mock.py
├── requirements_mock.txt           ← CPU-only deps
└── requirements.txt                ← full CUDA deps (includes mock)
```

Classifier weights (`.pth`) go in `models/.models/`:
```
models/.models/
├── densenet121.pth
├── densenet121_dct.pth
├── resnet50.pth
└── vit_b_16.pth
```

---

## 🖥️ Environment Setup

### Option A — CPU / mock mode (any machine, no GPU required)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements_mock.txt \
            --extra-index-url https://download.pytorch.org/whl/cpu
```

### Option B — Full mode (CUDA-capable machine)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

`torch` from PyPI automatically installs the CUDA-enabled build when CUDA drivers are present — no extra flags needed.

---

## 📐 Evaluating Adversarial Robustness

`evaluate.py` takes a set of original images and their adversarial counterparts, runs all classifiers, and produces an SSIM-weighted attack success score.

**1. Edit `scripts/config.yaml`** to point to your directories:

```yaml
original_root: test_set_deepfake/Dataset/test/fake
adv_root:      adversarial_dataset/Dataset/test/fake
models_dir:    ../.models           # folder containing *.pth weight files
classifiers:
  - resnet50
  - densenet121
  - vit_b_16
  - densenet121_dct
save_json: results.json
```

**2. Run:**

```bash
cd scripts
python3 evaluate.py --config config.yaml
```

Output printed to stdout and optionally saved as JSON (`save_json` key in config).

---

## 🔁 Feedback Loop — Fine-tune SD to Evade Classifiers

The feedback loop runs in **three phases**:

```
PHASE 1 — Pre-training evaluation
  Generate eval_iterations images → run each through the evaluators
  → print per-model fake detection rate

PHASE 2 — Training
  Generate images → backpropagate classifier evasion loss → update UNet weights
  (only the classifiers listed under 'classifiers' drive the gradient)

PHASE 3 — Post-training evaluation
  Same as Phase 1 → compare Before / After detection rates per evaluator
```

The distinction between `classifiers` and `evaluator` in the config is intentional:

| Config key | Role |
|---|---|
| `classifiers` | Provide the **gradient signal** during training. Use the models whose loss is differentiable (spatial: ResNet, DenseNet, ViT). |
| `evaluator` | **Measure** detection rate only — no gradient. Can include all models, including DCT-based ones. Defaults to `classifiers` if omitted. |
| `eval_iterations` | Number of images generated per evaluation pass (default `100`). |

At the end of a run, the terminal prints a summary like:

```
════════════════════════════════════════════════════════════════
  Training Effect Summary
════════════════════════════════════════════════════════════════
  Evaluator            Before      After         Δ
  ────────────────────────────────────────────────────────────
  resnet50              80.0%      60.0%   ▼20.0%
  densenet121          100.0%      80.0%   ▼20.0%
  vit_b_16              40.0%      40.0%   ─ 0.0%
  densenet121_dct        0.0%       0.0%   ─ 0.0%
  ────────────────────────────────────────────────────────────
  AGGREGATE             55.0%      45.0%   ▼10.0%

  Fake detection decreased by 10.0% after training.
  Evasion rate: 45.0% → 55.0%
════════════════════════════════════════════════════════════════
```

### Mock mode (CPU — verifies the full pipeline without any downloads)

Use this to validate the code before moving to a CUDA machine.

```bash
cd scripts
python3 feedback_loop_mock.py --config feedback_mock_config.yaml
```

Completes in ~1 second. Outputs go to `scripts/mock_output/`:

| File | Description |
|------|-------------|
| `step000X_imgY.png` | Decoded sample images saved every `save_images_every` steps |
| `unet_mock_stepXXXX.pth` | UNet checkpoints saved every `save_every` steps |
| `mock_training_history.json` | Pre/post evaluation results + per-step training loss |

### Real mode (CUDA — trains actual Stable Diffusion)

**1. Edit `scripts/feedback_config.yaml`:**

```yaml
sd_model_id: "runwayml/stable-diffusion-v1-5"

models_dir: ../../models/.models

# Training signal — gradient flows through these classifiers
classifiers:
  - resnet50
  - densenet121

# Measurement only — all models evaluated before and after training
evaluator:
  - resnet50
  - densenet121
  - vit_b_16
  - densenet121_dct

eval_iterations: 100       # images generated per evaluation pass

prompts:
  - "a photo of a person"
  - "a realistic portrait photograph"

image_height: 512
image_width:  512
batch_size: 1              # increase if VRAM > 12 GB
num_inference_steps: 20
total_iterations: 500
learning_rate: 1.0e-6
output_dir: feedback_output
```

**2. Run:**

```bash
cd scripts
python3 feedback_loop.py --config feedback_config.yaml
```

Outputs go to `feedback_output/`:

| File | Description |
|------|-------------|
| `step000X_imgY.png` | Sample images decoded every `save_images_every` steps |
| `unet_stepXXXX.pth` | UNet checkpoints saved every `save_every` steps |
| `unet_final.pth` | Final weights after all iterations |
| `training_history.json` | Pre/post evaluation results + full per-step training history |

**3. Load the trained UNet back into a pipeline:**

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.unet.load_state_dict(torch.load("feedback_output/unet_final.pth"))
pipe = pipe.to("cuda")

image = pipe("a photo of a person").images[0]
image.save("evading_output.png")
```

### Tuning tips

| Setting | Recommendation |
|---------|---------------|
| `learning_rate` | Start at `1e-6`; lower to `5e-7` if image quality degrades |
| `num_inference_steps` | `20` is fast; raise to `50` for higher quality samples |
| `batch_size` | `1` for 10 GB VRAM; `2` for 16 GB+ |
| `total_iterations` | `200–500` for initial experiments; `1000+` for full training |
| `eval_iterations` | `100` gives stable detection rate estimates; lower to `20` for quick checks |
| `save_every` | Keep low (`50`) so you can roll back if training diverges |

---

## 🏆 Results & Rankings

The challenge ended with strong global participation. Here are the final standings:

| Rank | Team Name | Organization/Institution | Final Score |
|------|-----------|-------------------------|-------------|
| 🥇 1st | **MR-CAS** | 🇨🇳 University of Chinese Academy of Sciences | **2740** |
| 🥈 2nd | **Safe AI** | 🇰🇷 UNIST (Ulsan National Institute of Science and Technology) | **2709** |
| 🥉 3rd | **RoMa** | 🇩🇪 Fraunhofer SIT \| ATHENE Center | **2679** |
| 4th | GRADIANT | 🇪🇸 Gradiant | 2631 |
| 5th | DASH | 🇰🇷 Sungkyunkwan University | 2618 |
| 6th | SecureML | 🇮🇹 University of Cagliari | 2490 |
| 7th | MICV | 🇨🇳 Ant Group | 2434 |
| 8th | WHU_PB | 🇨🇳 Wuhan University | 2354 |
| 9th | The Adversaries | 🇸🇬 Singapore Institute of Technology | 2341 |
| 10th | DeFakePol | 🇵🇱 Samsung Research Poland | 1665 |
| 11th | False Negative | 🇨🇳 The Hong Kong Polytechnic University | 1602 |
| 12th | VYAKRITI 2.0 | 🇮🇳 Apex Institute of technology Chandigarh University | 1041 |
| 13th | MILab | 🇨🇳 University of Science and Technology of China | 110 |

## 📊 Timeline

The AADD-2025 Challenge followed this timeline:

- ✅ **March 03, 2025**: Competition Website Launch  
- ✅ **March 15 - May 22, 2025**: Registration Period (Extended)
- ✅ **April 10, 2025**: Test Set and Classificator Release  
- ✅ **June 15, 2025**: Final Submission Deadline  
- ✅ **June 22, 2025**: Leaderboard Publication and Rankings Release
- ⏳ **June 30, 2025**: Paper Submission Deadline (Top 3 Teams Only)
- ⏳ **July 24, 2025**: Announcement regarding full paper submission
- ⏳ **August 01, 2025**: Camera ready - Grand Challenge Solutions (Top 3 Teams Only)
- ⏳ **ACM Multimedia 2025**: Conference & Winners Recognition

## 📝 Publication Opportunities

The top 3 teams were invited to submit full-length papers describing their methods in detail. These papers underwent a rigorous review process managed by the challenge organizers, with accepted papers included in the ACM Multimedia 2025 proceedings.

## 👥 Organizing Committee

### Chairs
| Name | Role | Email | Affiliation |
|------|------|-------|-------------|
| **Luca Guarnera** | Research Fellow | luca.guarnera@unict.it | Department of Mathematics and Computer Science, University of Catania, Italy |
| **Francesco Guarnera** | Research Fellow | francesco.guarnera@unict.it | Department of Mathematics and Computer Science, University of Catania, Italy |

### Co-Chairs
| Name | Role | Email | Affiliation |
|------|------|-------|-------------|
| **Sebastiano Battiato** | Full Professor | sebastiano.battiato@unict.it | Department of Mathematics and Computer Science, University of Catania, Italy |
| **Giovanni Puglisi** | Associate Professor | puglisi@unica.it | Department of Mathematics and Informatics, University of Cagliari, Italy |
| **Zahid Akhtar** | Associate Professor | akhtarz@sunypoly.edu | State University of New York Polytechnic Institute, USA |

### Technical Committee
| Name | Role | Email | Affiliation |
|------|------|-------|-------------|
| **Mirko Casu** | PhD Student | mirko.casu@phd.unict.it | Department of Mathematics and Computer Science, University of Catania, Italy |
| **Orazio Pontorno** | PhD Student | orazio.pontorno@phd.unict.it | Department of Mathematics and Computer Science, University of Catania, Italy |
| **Claudio Vittorio Ragaglia** | PhD Student | claudio.ragaglia@phd.unict.it | Department of Mathematics and Computer Science, University of Catania, Italy |

## 📧 Contact Information

**Main Contact**: Mirko Casu  
**Email**: challenge.dff@gmail.com 

## 📖 Citation

**Dataset Attribution**: Part of this challenge dataset is based on the WILD dataset. If you use the data, please also cite:

```bibtex
@misc{bongini2025wildnewinthewildimage,
      title={WILD: a new in-the-Wild Image Linkage Dataset for synthetic image attribution}, 
      author={Pietro Bongini and Sara Mandelli and Andrea Montibeller and Mirko Casu and Orazio Pontorno and Claudio Vittorio Ragaglia and Luca Zanchetta and Mattia Aquilina and Taiba Majid Wani and Luca Guarnera and Benedetta Tondi and Giulia Boato and Paolo Bestagini and Irene Amerini and Francesco De Natale and Sebastiano Battiato and Mauro Barni},
      year={2025},
      eprint={2504.19595},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2504.19595}, 
}
```

## 🌐 Related Resources

- [ACM Multimedia 2025 Conference](https://acmmm2025.org/)
- [Challenge Website](https://iplab.dmi.unict.it/mfs/acm-aadd-challenge-2025/)

**Institutional Affiliations:**
- **University of Catania** - [Department of Mathematics and Computer Science](https://web.dmi.unict.it/en)
- **University of Cagliari** - [Department of Mathematics and Informatics](https://web.unica.it/unica/en/dip_matinfo.page)
- **State University of New York Polytechnic Institute** - [Website](https://sunypoly.edu/)

© 2025 [University of Catania](https://www.unict.it/en).  
Powered by the [Multimedia Security and Forensics](https://iplab.dmi.unict.it/mfs/) group of the [Image Processing Laboratory (IPLAB)](https://iplab.dmi.unict.it/).

## 🏷️ Keywords

`Deepfake Detection`, `Adversarial Attacks`, `Computer Vision`, `Digital Forensics`, `AI Security`, `Media Authentication`, `Challenge Competition`, `ACM Multimedia`

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/assets/images/loghiunictiplab2.png">
  <source media="(prefers-color-scheme: light)" srcset="/assets/images/loghiunictiplabblack.png">
  <img alt="Logos" src="/assets/images/loghiunictiplab2.png">
</picture>
