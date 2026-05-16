# CNN Feature Analysis with LoRA

A study comparing pre-trained CNN and Vision Transformer feature representations for image retrieval on the Oxford-IIIT Pet dataset, before and after parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation).

## Overview

This project investigates how LoRA fine-tuning affects the quality of visual features extracted from four ImageNet pre-trained architectures. Feature quality is measured via **k-NN image retrieval** (Precision@10) on the 37-class pet classification dataset.

**Models supported:**

- ResNet-101
- GoogLeNet
- ViT-B/16
- MobileNet V2

## Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 37 breeds of cats and dogs (~7,400 images). Downloaded automatically on first run.

## Project Structure

```text
cnn_feature_analysis/
├── main.py             # CLI entry point
├── models.py           # FeatureExtractor with LoRA support
├── train.py            # LoRA fine-tuning loop
├── evaluate.py         # Feature extraction, k-NN eval, visualization
├── dataset.py          # Dataset loading and transforms
├── requirements.txt
├── data/               # Auto-downloaded Oxford-IIIT Pet dataset
│   └── oxford-iiit-pet/
├── models/             # Saved LoRA adapter weights (per model)
│   └── <model>_lora/
└── results/            # Retrieval visualization plots
    └── <model>_<mode>_query_<i>.png
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

All workflows are run through `main.py`:

### 1. Baseline evaluation (pre-trained, no fine-tuning)

```bash
python main.py --mode baseline --model resnet101
```

### 2. Fine-tune with LoRA

```bash
python main.py --mode train --model resnet101 --epochs 5 --lora_r 8
```

Trained adapters are saved to `./models/<model>_lora/`.

### 3. Evaluate after LoRA fine-tuning

```bash
python main.py --mode evaluate_lora --model resnet101
```

### CLI arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--mode` | required | `baseline`, `train`, or `evaluate_lora` |
| `--model` | `resnet101` | `resnet101`, `googlenet`, `vit_b_16`, `mobilenet_v2` |
| `--epochs` | `5` | Training epochs |
| `--lora_r` | `4` | LoRA rank |
| `--batch_size` | `32` | Batch size |
| `--output_dir` | `./results` | Where to save retrieval visualizations |
| `--model_dir` | `./models` | Where to save/load LoRA adapters |

## How It Works

**Feature extraction** — forward hooks capture activations from the penultimate layer of each model (before the classification head), producing a fixed-size embedding per image.

**LoRA application** — `peft` applies low-rank adapters to targeted convolutional or attention layers:

- ResNet-101: all 3×3 convolutions in bottleneck blocks
- GoogLeNet: all `conv` layers inside `BasicConv2d` modules
- ViT-B/16: query and value projection matrices in attention layers
- MobileNet V2: expand (1×1) pointwise convolutions

**Evaluation** — L2-normalized embeddings are indexed with a ball-tree k-NN. Precision@10 measures what fraction of the 10 nearest neighbors share the query image's breed label.

**Visualization** — retrieval results for random query images are saved as plots with green/red borders indicating correct/incorrect neighbors.

## Hardware

Automatically uses CUDA > Apple MPS > CPU.
