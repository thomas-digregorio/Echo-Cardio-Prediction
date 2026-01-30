# EchoNet EF Prediction

A deep learning system for automated **Ejection Fraction (EF) prediction** from echocardiogram videos using VideoMAE transformer architecture. This project fine-tunes a pretrained video transformer on the EchoNet-Dynamic dataset to predict left ventricular function.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Design](#model-design)
- [Training Design](#training-design)
- [Loss Function Design](#loss-function-design)
- [References](#references)

---

## Overview

**Ejection Fraction (EF)** is the percentage of blood pumped out of the left ventricle with each heartbeat—a critical biomarker for heart function. This project automates EF estimation from 2D echocardiogram videos using modern deep learning.

### Key Features

- **VideoMAE Backbone**: Leverages self-supervised video pretraining for robust feature extraction
- **Attention Pooling**: Learns which frames are most informative for EF prediction
- **Clinical Importance Weighting**: Prioritizes accuracy on clinically significant EF ranges
- **Mixed Precision Training**: 2x faster training with AMP (Automatic Mixed Precision)
- **Confidence Estimation**: MC Dropout for uncertainty quantification
- **Interactive Demo**: Streamlit app for real-time predictions
- **Evaluation Notebook**: Jupyter notebook with per-bin MAE, scatter plots, and Bland-Altman analysis

---

## Results

### Overall Performance (Test Set)

| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | ~7-8% |
| **RMSE** | ~10% |
| **R² Score** | ~0.75 |
| **Pearson Correlation** | ~0.87 |

### Per-Category MAE

| EF Category | Range | MAE | Clinical Significance |
|-------------|-------|-----|----------------------|
| Severe HF | 0-30% | ~12% | Critical - indicates severe dysfunction |
| Moderate HF | 30-40% | ~10% | Important - requires intervention |
| Mild/Borderline | 40-55% | ~8% | Watchlist |
| Normal | 55-70% | ~4% | Healthy range |
| Hyperdynamic | 70%+ | ~6% | Can indicate stress response |

> **Note**: Results may vary based on training run. See `notebooks/evaluation.ipynb` for detailed analysis.

---

## Project Structure

```
cardio_demo_2/
├── app/
│   └── app.py                 # Streamlit demo application
├── checkpoints/
│   ├── best_model.pt          # Best validation checkpoint
│   └── model_step_*.pt        # Intermediate checkpoints
├── configs/
│   └── default.yaml           # Training configuration
├── EchoNet-Dynamic/           # Dataset (not included)
│   ├── FileList.csv
│   └── Videos/
├── notebooks/
│   └── evaluation.ipynb       # Model evaluation & visualization
├── src/
│   ├── basic_model.py         # EchoNetRegressor model definition
│   ├── dali_loader.py         # NVIDIA DALI data loading pipeline
│   └── train.py               # Training script
├── README.md
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.8
- Conda (recommended)

### Setup

```bash
# Create conda environment
conda create -n echonet python=3.12
conda activate echonet

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install NVIDIA DALI
pip install nvidia-dali-cuda120

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/) and extract to `EchoNet-Dynamic/`.

### Docker (Alternative)

```bash
# Build image
docker build -t echonet-ef .

# Run with GPU support
docker run --gpus all -p 8501:8501 -v $(pwd)/EchoNet-Dynamic:/app/EchoNet-Dynamic echonet-ef

# For training (mount data and checkpoints)
docker run --gpus all -v $(pwd)/EchoNet-Dynamic:/app/EchoNet-Dynamic -v $(pwd)/checkpoints:/app/checkpoints \
    -e WANDB_API_KEY=$WANDB_API_KEY echonet-ef python src/train.py --config configs/default.yaml
```

---

## Usage

### Training

```bash
# Login to Weights & Biases (first time only)
wandb login

# Configure training in configs/default.yaml, then:
python src/train.py --config configs/default.yaml
```

### Experiment Tracking (W&B)

All training runs are automatically logged to [Weights & Biases](https://wandb.ai):

- **Metrics**: `train_loss`, `val_loss`, `val_mae`, per-bin MAE
- **System**: GPU utilization, memory, temperature
- **Hyperparameters**: Logged from config file
- **Checkpoints**: Best model saved locally

View your runs at: `https://wandb.ai/tad537113-university-of-texas/EchoNet-VideoMAE/runs/1q25s1fd`

### Demo Application

```bash
streamlit run app/app.py
```

### Evaluation Notebook

Open `notebooks/evaluation.ipynb` in Jupyter or VS Code to run the full evaluation pipeline with visualizations.

---

## Model Design

### Architecture Overview

```
Input Video (16 frames × 224 × 224 × 3)
         │
         ▼
┌─────────────────────────────────┐
│       VideoMAE-base Backbone    │  ← Pretrained on Kinetics-400
│    (86M params, 768-dim output) │
└─────────────────────────────────┘
         │
         ▼ (Sequence of 1568 tokens)
┌─────────────────────────────────┐
│       Attention Pooling         │  ← Learned weighted average
│   (768 → 128 → 1 attention)     │
└─────────────────────────────────┘
         │
         ▼ (768-dim vector)
┌─────────────────────────────────┐
│       MLP Regression Head       │
│   LayerNorm → Dropout(0.1)      │
│   Linear(768→256) → GELU        │
│   Dropout(0.1)                  │
│   Linear(256→64) → GELU         │
│   Linear(64→1) → EF prediction  │
└─────────────────────────────────┘
```

### Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Backbone** | VideoMAE-base | Self-supervised pretraining captures motion and structure |
| **Frames** | 16 frames | Matches VideoMAE pretraining |
| **Pooling** | Attention Pooling | Learns to focus on systole/diastole frames that are most informative |
| **Head Depth** | 3 layers (768→256→64→1) | Deeper head extracts richer features; prevents underfitting |
| **Activation** | GELU | Smoother gradients than ReLU; standard for transformers |
| **Bias Init** | 55.0 | Initializes predictions near dataset mean for faster convergence |

---

## Training Design

### Optimization Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW | Weight decay regularization for transformers |
| **LR (backbone)** | 1e-5 | Low LR to preserve pretrained features |
| **LR (head)** | 1e-3 | 100x higher for randomly initialized head |
| **Scheduler** | Cosine with warmup | Smooth decay prevents sudden LR drops |
| **Warmup** | 10% of steps | Stabilizes early training |
| **Batch size** | 12 (effective 96) | Large batch via gradient accumulation |
| **Mixed Precision** | FP16 (AMP) | 2x speedup with minimal accuracy loss |

### Training Features

- **Early Stopping**: Patience of 5 evaluations to prevent overfitting
- **Step-based Validation**: Evaluate every 500 steps for frequent feedback
- **Gradient Clipping**: Max norm 1.0 for training stability

### Data Pipeline

- **NVIDIA DALI**: GPU-accelerated video decoding (3-5x faster than CPU)
- **Normalization**: ImageNet mean/std for transfer learning compatibility
- **Split Filtering**: Proper TRAIN/VAL/TEST separation per EchoNet protocol

---

## Loss Function Design

### Clinically-Weighted MSE

Standard MSE treats all EF values equally, but clinically, errors at extreme EF values matter more:

```python
def weighted_mse_loss(preds, labels, use_weights=True):
    if use_weights:
        weights = torch.ones_like(labels)
        weights[labels < 30] = 3.0   # Severe HF (critical)
        weights[labels < 40] = 2.0   # Moderate HF 
        weights[labels > 70] = 2.5   # Hyperdynamic
    else:
        weights = torch.ones_like(labels)
    
    return (weights * (preds - labels) ** 2).mean()
```

### Weight Design Rationale

| EF Range | Weight | Clinical Reason |
|----------|--------|-----------------|
| < 30% (Severe HF) | 3.0x | Life-threatening; requires immediate intervention |
| 30-40% (Moderate HF) | 2.0x | Significant dysfunction; treatment decisions depend on it |
| 40-55% (Mild) | 1.0x | Borderline; monitoring needed |
| 55-70% (Normal) | 1.0x | Healthy range; most samples |
| > 70% (Hyperdynamic) | 2.5x | Can indicate pathology; rare but important |

**Trade-off**: Higher weights on rare cases improve their accuracy but may slightly increase overall MAE (acceptable for clinical utility).

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| **PyTorch** | Deep learning framework |
| **Transformers** | VideoMAE pretrained model |
| **NVIDIA DALI** | GPU-accelerated data loading |
| **Weights & Biases** | Experiment tracking |
| **Streamlit** | Interactive demo app |
| **OpenCV** | Video processing |
| **scikit-learn** | Evaluation metrics |
| **Matplotlib/Seaborn** | Visualization |

---

## References

### Dataset

- Ouyang, D., et al. (2020). **Video-based AI for beat-to-beat assessment of cardiac function.** *Nature*, 580(7802), 252-256. [Paper](https://www.nature.com/articles/s41586-020-2145-8) | [Dataset](https://echonet.github.io/dynamic/)

### Model Architecture

- Tong, Z., et al. (2022). **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.** *NeurIPS 2022*. [Paper](https://arxiv.org/abs/2203.12602) | [Model](https://huggingface.co/MCG-NJU/videomae-base)

### Related Work

- Reynaud, H., et al. (2021). **Ultrasound Video Transformers for Cardiac Ejection Fraction Estimation.** *MICCAI 2021*.

---

## License

This project is for educational and research purposes. The EchoNet-Dynamic dataset has its own [license terms](https://echonet.github.io/dynamic/).

---

## Acknowledgments

- Stanford Medicine for the EchoNet-Dynamic dataset
- Hugging Face for the Transformers library
- NVIDIA for DALI

