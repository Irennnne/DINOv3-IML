# DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for Image Forensics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

> **TL;DR** — Freeze DINOv3, inject LoRA on QKV, attach a 3-conv head. With only **9.1 M trainable parameters**, this simple recipe outperforms all prior specialized detectors on both the CAT-Net and MVSS-Net evaluation protocols.

<p align="center">
  <img src="assets/teaser_input.jpg" width="200" alt="Tampered input"/>
  &nbsp;&nbsp;→&nbsp;&nbsp;
  <img src="assets/teaser_mask.png" width="200" alt="Predicted manipulation mask"/>
</p>
<p align="center"><em>Left: tampered image. Right: predicted forgery mask.</em></p>

---

## Highlights

- **State-of-the-art** avg pixel-F1 on CAT-Net protocol: **0.847** (vs. 0.677 prior SOTA)
- Only **9.1 M trainable parameters** — LoRA on QKV + 3-conv head
- **Frozen backbone** — no catastrophic forgetting, no collapse on small datasets
- **Simple architecture** — no specialized forensic components, no frequency analysis, no attention manipulation

---

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="800" alt="Architecture diagram"/>
</p>

A frozen DINOv3 ViT backbone with LoRA injected on QKV projections produces dense patch tokens, which are reshaped into a 2D feature map and decoded by a lightweight convolutional segmentation head into a pixel-level manipulation mask. The decoder applies Conv3×3→BN→ReLU (×2), then Conv1×1→Sigmoid, followed by bilinear upsampling to the original input resolution.

---

## Results

### CAT-Net Protocol (4-dataset avg pixel-F1)

*Training: CASIA-v2 + FantasticReality + IMD2020 + TampCOCO. Test: CASIAv1 / Columbia / NIST16 / Coverage.*

| | Method | CASIAv1 | Columbia | NIST16 | Coverage | **Avg F1** |
|---|---|---|---|---|---|---|
| Prior | MVSS-Net | 0.583 | 0.740 | 0.336 | 0.486 | 0.536 |
| Prior | PSCC-Net | 0.630 | 0.884 | 0.346 | 0.448 | 0.577 |
| Prior | CAT-Net | 0.808 | **0.915** | 0.252 | 0.427 | 0.601 |
| Prior | TruFor | 0.818 | 0.885 | 0.348 | 0.457 | 0.627 |
| Prior | Mesorch | **0.840** | 0.890 | **0.392** | **0.586** | **0.677** |
| ViT-S | DINOv3 + LoRA r=32 | 0.787 | 0.923 | 0.462 | 0.646 | 0.704 |
| ViT-S | DINOv3 + LoRA r=64 | 0.803 | 0.918 | 0.457 | 0.671 | 0.712 |
| ViT-S | DINOv3 + Full FT | 0.704 | 0.887 | 0.400 | 0.531 | 0.630 |
| ViT-B | DINOv3 + LoRA r=32 | 0.840 | 0.938 | 0.565 | 0.715 | 0.764 |
| ViT-B | DINOv3 + LoRA r=64 | 0.863 | 0.904 | 0.570 | 0.784 | 0.780 |
| ViT-B | DINOv3 + Full FT | 0.815 | 0.918 | 0.518 | 0.652 | 0.726 |
| ViT-L | DINOv3 + LoRA r=32 | 0.907 | **0.941** | **0.636** | **0.905** | **0.847** |
| ViT-L | DINOv3 + LoRA r=64 | **0.908** | 0.927 | 0.633 | 0.882 | 0.837 |
| ViT-L | DINOv3 + Full FT | 0.882 | 0.938 | 0.616 | 0.866 | 0.826 |

### MVSS-Net Protocol (5-dataset avg pixel-F1)

*Training: CASIA-v2 only (5,123 images). Test: CASIAv1 / Columbia / NIST16 / Coverage / IMD2020.*

| | Method | Coverage | Columbia | NIST16 | CASIAv1 | IMD2020 | **Avg F1** |
|---|---|---|---|---|---|---|---|
| Prior | Mantra-Net | 0.090 | 0.243 | 0.104 | 0.125 | 0.055 | 0.123 |
| Prior | ObjectFormer | 0.294 | 0.336 | 0.173 | 0.429 | 0.173 | 0.281 |
| Prior | PSCC-Net | 0.231 | 0.604 | 0.214 | 0.378 | 0.235 | 0.333 |
| Prior | NCL-IML | 0.225 | 0.446 | 0.260 | 0.502 | 0.237 | 0.334 |
| Prior | MVSS-Net | 0.259 | 0.386 | 0.246 | 0.534 | 0.279 | 0.341 |
| Prior | CAT-Net | 0.296 | 0.584 | 0.269 | 0.581 | 0.273 | 0.401 |
| Prior | IML-ViT | **0.435** | 0.780 | 0.331 | **0.721** | **0.327** | 0.519 |
| Prior | TruFor | 0.419 | **0.865** | **0.324** | **0.721** | 0.322 | **0.530** |
| ViT-S | DINOv3 + LoRA r=32 | 0.363 | 0.646 | 0.324 | 0.622 | 0.346 | 0.460 |
| ViT-S | DINOv3 + LoRA r=64 | 0.403 | 0.721 | 0.343 | 0.672 | 0.373 | 0.502 |
| ViT-S | DINOv3 + Full FT | 0.154 | 0.364 | 0.189 | 0.209 | 0.186 | 0.221 |
| ViT-B | DINOv3 + LoRA r=32 | 0.211 | 0.443 | 0.256 | 0.543 | 0.296 | 0.350 |
| ViT-B | DINOv3 + LoRA r=64 | 0.545 | 0.820 | 0.465 | 0.761 | 0.475 | 0.613 |
| ViT-B | DINOv3 + Full FT | 0.163 | 0.136 | 0.093 | 0.078 | 0.082 | 0.110 |
| ViT-L | DINOv3 + LoRA r=32 | **0.822** | **0.943** | 0.589 | 0.867 | 0.628 | 0.770 |
| ViT-L | DINOv3 + LoRA r=64 | 0.820 | 0.915 | **0.621** | **0.873** | **0.641** | **0.774** |
| ViT-L | DINOv3 + Full FT | 0.679 | 0.842 | 0.532 | 0.852 | 0.499 | 0.681 |

### Robustness Evaluation

We evaluate robustness on CASIAv1 under three post-processing perturbations using models trained under the CAT-Net protocol.

**Gaussian Noise** (variance 3–23):

| | Method | Clean | 3 | 7 | 11 | 15 | 19 | 23 | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Prior | MVSS-Net | 0.583 | 0.582 | 0.582 | 0.576 | 0.574 | 0.562 | 0.561 | 0.574 |
| Prior | PSCC-Net | 0.630 | 0.613 | 0.575 | 0.554 | 0.540 | 0.523 | 0.512 | 0.564 |
| Prior | CAT-Net | 0.808 | 0.798 | 0.788 | 0.783 | 0.772 | 0.757 | 0.755 | 0.780 |
| Prior | TruFor | 0.821 | 0.767 | 0.738 | 0.719 | 0.695 | 0.683 | 0.678 | 0.729 |
| Prior | Mesorch | 0.840 | 0.821 | 0.805 | 0.797 | 0.789 | 0.778 | 0.770 | 0.800 |
| ViT-S | DINOv3 + LoRA r=32 | 0.787 | 0.784 | 0.772 | 0.770 | 0.758 | 0.756 | 0.752 | 0.768 |
| ViT-S | DINOv3 + LoRA r=64 | 0.803 | 0.798 | 0.790 | 0.787 | 0.775 | 0.775 | 0.769 | 0.785 |
| ViT-S | DINOv3 + Full FT | 0.704 | 0.700 | 0.695 | 0.692 | 0.689 | 0.682 | 0.679 | 0.692 |
| ViT-B | DINOv3 + LoRA r=32 | 0.840 | 0.830 | 0.815 | 0.806 | 0.801 | 0.796 | 0.787 | 0.811 |
| ViT-B | DINOv3 + LoRA r=64 | 0.863 | 0.849 | 0.838 | 0.826 | 0.820 | 0.814 | 0.815 | 0.832 |
| ViT-B | DINOv3 + Full FT | 0.815 | 0.807 | 0.795 | 0.783 | 0.779 | 0.775 | 0.772 | 0.789 |
| ViT-L | DINOv3 + LoRA r=32 | 0.907 | **0.903** | **0.901** | **0.899** | **0.897** | **0.889** | **0.892** | **0.898** |
| ViT-L | DINOv3 + LoRA r=64 | **0.908** | 0.897 | 0.896 | 0.892 | 0.890 | 0.888 | 0.887 | 0.894 |
| ViT-L | DINOv3 + Full FT | 0.882 | 0.878 | 0.876 | 0.873 | 0.870 | 0.870 | 0.868 | 0.874 |

**Gaussian Blur** (kernel size 3–23):

| | Method | Clean | 3 | 7 | 11 | 15 | 19 | 23 | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Prior | MVSS-Net | 0.583 | 0.459 | 0.310 | 0.237 | 0.189 | 0.157 | 0.139 | 0.296 |
| Prior | PSCC-Net | 0.630 | 0.541 | 0.453 | 0.316 | 0.166 | 0.114 | 0.078 | 0.328 |
| Prior | CAT-Net | 0.808 | 0.751 | 0.653 | 0.544 | 0.434 | 0.314 | 0.214 | 0.531 |
| Prior | TruFor | 0.821 | 0.751 | 0.688 | 0.603 | 0.456 | 0.274 | 0.130 | 0.532 |
| Prior | Mesorch | 0.840 | 0.790 | 0.708 | 0.628 | 0.533 | 0.419 | 0.294 | 0.602 |
| ViT-S | DINOv3 + LoRA r=32 | 0.787 | 0.740 | 0.669 | 0.582 | 0.453 | 0.334 | 0.253 | 0.545 |
| ViT-S | DINOv3 + LoRA r=64 | 0.803 | 0.761 | 0.697 | 0.612 | 0.486 | 0.370 | 0.294 | 0.575 |
| ViT-S | DINOv3 + Full FT | 0.704 | 0.645 | 0.558 | 0.443 | 0.314 | 0.234 | 0.162 | 0.437 |
| ViT-B | DINOv3 + LoRA r=32 | 0.840 | 0.806 | 0.765 | 0.684 | 0.546 | 0.389 | 0.270 | 0.614 |
| ViT-B | DINOv3 + LoRA r=64 | 0.863 | 0.836 | 0.787 | 0.712 | 0.529 | 0.309 | 0.185 | 0.603 |
| ViT-B | DINOv3 + Full FT | 0.815 | 0.766 | 0.711 | 0.622 | 0.495 | 0.365 | 0.263 | 0.577 |
| ViT-L | DINOv3 + LoRA r=32 | 0.907 | **0.886** | **0.848** | **0.807** | **0.722** | **0.590** | **0.479** | **0.748** |
| ViT-L | DINOv3 + LoRA r=64 | **0.908** | 0.880 | 0.837 | 0.794 | 0.667 | 0.481 | 0.356 | 0.703 |
| ViT-L | DINOv3 + Full FT | 0.882 | 0.855 | 0.818 | 0.754 | 0.628 | 0.452 | 0.322 | 0.673 |

**JPEG Compression** (quality 100–50):

| | Method | Clean | 100 | 90 | 80 | 70 | 60 | 50 | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Prior | MVSS-Net | 0.583 | 0.570 | 0.545 | 0.517 | 0.491 | 0.449 | 0.389 | 0.506 |
| Prior | PSCC-Net | 0.630 | 0.622 | 0.579 | 0.493 | 0.452 | 0.385 | 0.287 | 0.493 |
| Prior | CAT-Net | 0.808 | 0.790 | 0.786 | 0.743 | 0.723 | 0.684 | 0.613 | 0.735 |
| Prior | TruFor | 0.821 | 0.806 | 0.794 | 0.702 | 0.685 | 0.633 | 0.494 | 0.705 |
| Prior | Mesorch | 0.840 | 0.831 | 0.819 | 0.772 | 0.771 | 0.729 | 0.655 | 0.774 |
| ViT-S | DINOv3 + LoRA r=32 | 0.787 | 0.777 | 0.765 | 0.721 | 0.708 | 0.669 | 0.582 | 0.716 |
| ViT-S | DINOv3 + LoRA r=64 | 0.803 | 0.795 | 0.782 | 0.746 | 0.729 | 0.686 | 0.598 | 0.734 |
| ViT-S | DINOv3 + Full FT | 0.704 | 0.692 | 0.678 | 0.629 | 0.603 | 0.560 | 0.486 | 0.622 |
| ViT-B | DINOv3 + LoRA r=32 | 0.840 | 0.832 | 0.819 | 0.785 | 0.766 | 0.736 | 0.658 | 0.776 |
| ViT-B | DINOv3 + LoRA r=64 | 0.863 | 0.855 | 0.846 | 0.812 | 0.798 | 0.784 | 0.710 | 0.810 |
| ViT-B | DINOv3 + Full FT | 0.815 | 0.806 | 0.794 | 0.757 | 0.743 | 0.711 | 0.629 | 0.751 |
| ViT-L | DINOv3 + LoRA r=32 | 0.907 | **0.906** | 0.895 | 0.876 | 0.875 | 0.860 | **0.806** | 0.875 |
| ViT-L | DINOv3 + LoRA r=64 | **0.908** | 0.904 | **0.896** | **0.878** | **0.876** | **0.862** | 0.805 | **0.875** |
| ViT-L | DINOv3 + Full FT | 0.882 | 0.878 | 0.875 | 0.849 | 0.846 | 0.831 | 0.772 | 0.847 |

---

## Pretrained Weights

| Config | Protocol | Avg F1 | Trainable params | Download |
|---|---|---|---|---|
| ViT-L LoRA r=32 | CAT | **0.847** | 9.1 M | [Google Drive](https://drive.google.com/drive/folders/1X9EZUnvg7FLBvGgR_V7mViZUhiD5r7YT) |
| ViT-L LoRA r=64 | CAT | 0.837 | 12.2 M | [Google Drive](https://drive.google.com/drive/folders/1X9EZUnvg7FLBvGgR_V7mViZUhiD5r7YT) |
| ViT-L LoRA r=64 | MVSS | **0.774** | 12.2 M | [Google Drive](https://drive.google.com/drive/folders/1X9EZUnvg7FLBvGgR_V7mViZUhiD5r7YT) |
| ViT-L LoRA r=32 | MVSS | 0.770 | 9.1 M | [Google Drive](https://drive.google.com/drive/folders/1X9EZUnvg7FLBvGgR_V7mViZUhiD5r7YT) |
| ViT-B LoRA r=64 | CAT | 0.780 | 5.7 M | [Google Drive](https://drive.google.com/drive/folders/1X9EZUnvg7FLBvGgR_V7mViZUhiD5r7YT) |
| ViT-S LoRA r=32 | CAT | 0.704 | 1.4 M | [Google Drive](https://drive.google.com/drive/folders/1X9EZUnvg7FLBvGgR_V7mViZUhiD5r7YT) |

DINOv3 backbone weights: see the [DINOv3 / DINOv2 repository](https://github.com/facebookresearch/dinov2).

---

## Quick Start — Inference

```bash
pip install torch peft Pillow numpy
```

```python
from inference import predict

mask = predict(
    image_path="path/to/image.jpg",
    checkpoint_path="checkpoints/cat_vitl_lora_r32.pth",
    dinov3_repo="path/to/dinov3",
    dinov3_weights="path/to/dinov3_vitl16_pretrain.pth",
    model_type="dinov3_vitl16",
    lora_rank=32,
)
mask.save("predicted_mask.png")
```

Or via command line:

```bash
python inference.py \
    --image photo.jpg \
    --checkpoint checkpoints/cat_vitl_lora_r32.pth \
    --dinov3_repo path/to/dinov3 \
    --dinov3_weights path/to/dinov3_vitl16_pretrain.pth \
    --model_type dinov3_vitl16 \
    --lora_rank 32 \
    --output mask.png
```

---

## Training

### 1. Install dependencies

```bash
pip install torch peft imdlbenco
```

### 2. Download datasets

Follow [IMDLBenCo dataset preparation](https://github.com/scu-zjz/IMDLBenCo) to obtain CASIA-v2, FantasticReality, IMD2020, TampCOCO (CAT protocol) or CASIA-v2 alone (MVSS protocol).

### 3. Edit config

```bash
# Set data_path, test_data_path, dinov3_repo_path, dinov3_weights_path
vim configs/cat_lora_vitl_r32.yaml
```

Training hyperparameters (shared across all configs): AdamW optimizer, lr = 3e-4, cosine annealing schedule, 5 warmup epochs, 100 total epochs, effective batch size = 240 (via gradient accumulation), images resized to 512x512.

### 4. Launch training

```bash
# Single GPU
bash scripts/train.sh configs/cat_lora_vitl_r32.yaml

# Multi-GPU (e.g., 4 GPUs)
NPROC=4 bash scripts/train.sh configs/cat_lora_vitl_r32.yaml
```

Checkpoints and logs are saved to `output/<config_name>/`.

---

## Model Zoo (programmatic import)

The models can be imported and used without IMDLBenCo:

```python
from models import DINOv3ForensicsLoRA

model = DINOv3ForensicsLoRA(
    dinov3_repo_path="./dinov3",
    dinov3_weights_path="./dinov3_vitl16_pretrain.pth",
    dinov3_model_type="dinov3_vitl16",
    lora_rank=32,
    lora_alpha=64,
)

# Inference
import torch
image = torch.randn(1, 3, 512, 512)
mask = model.predict(image)   # (1, 1, 512, 512), values in [0, 1]

# Load from checkpoint
model = DINOv3ForensicsLoRA.from_pretrained(
    "checkpoints/cat_vitl_lora_r32.pth",
    dinov3_repo_path="./dinov3",
    dinov3_weights_path="./dinov3_vitl16_pretrain.pth",
    dinov3_model_type="dinov3_vitl16",
    lora_rank=32, lora_alpha=64,
)
```

---

## Citation

```bibtex
@article{yu2025dinov3iml,
  title   = {DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for Image Forensics},
  author  = {Yu, Jieming and Wang, Zhuohan and Ma, Xiaochen},
  journal = {arXiv preprint},
  year    = {2025},
}
```

---

## Acknowledgements

- [DINOv2 / DINOv3](https://github.com/facebookresearch/dinov2) (Facebook Research) for the pretrained ViT backbone
- [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) for the image manipulation detection training framework
- [PEFT](https://github.com/huggingface/peft) for the LoRA implementation
