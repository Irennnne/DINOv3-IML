# DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for Image Forensics

[![](https://img.shields.io/github/stars/Irennnne/DINOv3-IML/?style=social)](https://github.com/Irennnne/DINOv3-IML/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.16083-b31b1b.svg)](https://arxiv.org/abs/2604.16083)

> **TL;DR** — Freeze DINOv3, inject LoRA on QKV, attach a 3-conv head. With only **9.1 M trainable parameters**, this simple recipe outperforms all prior specialized detectors on both the CAT-Net and MVSS-Net evaluation protocols.

<p align="center">
  <img src="assets/teaser_input.jpg" width="200" alt="Tampered input"/>
  &nbsp;&nbsp;→&nbsp;&nbsp;
  <img src="assets/teaser_mask.png" width="200" alt="Predicted manipulation mask"/>
</p>
<p align="center"><em>Left: tampered image. Right: predicted forgery mask.</em></p>

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

---

## Pretrained Weights

### CAT-Net Protocol

| Config | Avg F1 | Trainable params | Download |
|---|---|---|---|
| ViT-S LoRA r=32 | 0.704 | 1.4 M | [Google Drive](https://drive.google.com/drive/folders/1YtHrQ9-j74OK3XzhYVsIKZZAodhcCNsS) |
| ViT-S LoRA r=64 | 0.712 | 2.0 M | [Google Drive](https://drive.google.com/drive/folders/1z--C_sdNFi4z7MTOnkT-Rjv7Bx99kwcT) |
| ViT-S Full FT   | 0.630 | ~22 M | [Google Drive](https://drive.google.com/drive/folders/1HgAHyNVeNfLYyUG-7xtJJvqixGBgXoDy) |
| ViT-B LoRA r=32 | 0.764 | 4.5 M | [Google Drive](https://drive.google.com/drive/folders/1pYOSBYGHg3Aby8MtvaXAXV5IhFdN7JYw) |
| ViT-B LoRA r=64 | 0.780 | 5.7 M | [Google Drive](https://drive.google.com/drive/folders/16c87obUh0VG0wzQP5lOWuEkjyLwo6toP) |
| ViT-B Full FT   | 0.726 | ~89 M | [Google Drive](https://drive.google.com/drive/folders/1A-0Mmf_vXDHeHnfzsuUoSjU9LZOk8Yf4) |
| ViT-L LoRA r=32 | **0.847** | 9.1 M | [Google Drive](https://drive.google.com/drive/folders/125leLub_M-lICa1ILTOL-FCz4ZY6eutj) |
| ViT-L LoRA r=64 | 0.837 | 12.2 M | [Google Drive](https://drive.google.com/drive/folders/1ygQjiQqtd2C7aGL6TdPh7hQC-bXw4k45) |
| ViT-L Full FT   | 0.826 | ~309 M | [Google Drive](https://drive.google.com/drive/folders/1PFpkmihTNFLdp5U3AFMVN92Sy3FxhNXd) |

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
# Pick a model config, then set:
#   - data_path / test_data_path   (CAT or MVSS)
#   - dinov3_repo_path
#   - dinov3_weights_path
vim configs/lora_vitl_r32.yaml
```

Configs are model-centric rather than protocol-specific. To switch between CAT and MVSS, edit only `data_path` and `test_data_path` in the YAML:

```yaml
# CAT
data_path: /path/to/balanced_dataset_cat.json
test_data_path: /path/to/test_datasets_cat.json

# MVSS
data_path: /path/to/casia2_tampered.json
test_data_path: /path/to/test_datasets_mvss.json
```

Training hyperparameters are shared through `_base/*.yaml`: AdamW optimizer, lr = 3e-4, cosine schedule, 5 warmup epochs, 100 total epochs, effective batch size via `accum_iter=10`, and 512x512 inputs.

### 4. Launch training

```bash
# Single GPU
bash scripts/train.sh configs/lora_vitl_r32.yaml

# Multi-GPU (e.g., 4 GPUs)
NPROC=4 bash scripts/train.sh configs/lora_vitl_r32.yaml
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
@misc{yu2026dinov3beatsspecializeddetectors,
  title={DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for Image Forensics},
  author={Jieming Yu and Qiuxiao Feng and Zhuohan Wang and Xiaochen Ma},
  year={2026},
  eprint={2604.16083},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.16083}
}
```

---

## Acknowledgements

- [DINOv2 / DINOv3](https://github.com/facebookresearch/dinov2) (Facebook Research) for the pretrained ViT backbone
- [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) for the image manipulation detection training framework
- [PEFT](https://github.com/huggingface/peft) for the LoRA implementation
