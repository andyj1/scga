# Improving Black-Box Generative Attacks via Generator Semantic Consistency

This repository contains the implementation of SCGA, a method to improve black-box adversarial transferability of generative attacks.

## Commands
```bash
python train_BIA_ours.py
```

📖 **Paper**: [MDD-VL: Multimodal Dataset Distillation with Vision-Language Models](https://arxiv.org/abs/????)

This repository contains the official implementation for multimodal dataset distillation using vision-language models with support for both image-text retrieval and zero-shot classification tasks.

<!-- <p align="center">
  <img src="imgs/figure.png" alt="problem" title="problem" width="700">
</p> -->

## ✨ Key Features

- 🚀 **Lightning Fabric** integration for efficient distributed training
- 🎯 **Dual Evaluation System**: Unified evaluation for both retrieval and classification tasks
- 🔄 **Hydra Configuration**: Flexible YAML-based configuration management
- 📊 **Multiple Datasets**: Support for Flickr8k/30k, COCO, CIFAR-10/100, ImageNet, and more
- 🎨 **Vision-Language Models**: CLIP-based architecture with customizable encoders
- ⚡ **Multi-GPU Support**: Distributed training with automatic device management

## 📋 Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Documentation](#documentation)
- [License](#license)

## 🔧 Requirements

- **Python**: >= 3.10
- **PyTorch**: 2.6.0 (with CUDA 11.8)
- **Lightning**: 2.5.5
- **CUDA**: 11.8 or higher (for GPU support)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/andyj1/SCGA.git
cd SCGA
```
### 2. Install Dependencies
```bash
conda create -f environment.yaml
```


## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{mdd-vl2025,
  title={Improving Black-Box Generative Attacks via Generator Semantic Consistency},
  author={Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon},
  journal={arxix:2506.18248},
  year={2025}
}
```
