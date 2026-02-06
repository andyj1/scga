# Improving Black-Box Generative Attacks via Generator Semantic Consistency (ICLR 2026)

This official repository contains the implementation of SCGA, a semantic structure-aware add-on method to improve black-box adversarial transferability of generative attacks.

## Commands
```bash
python train_BIA_ours.py
```

📖 **Paper**: [Improving Black-Box Generative Attacks via Generator Semantic Consistency]([https://arxiv.org/abs/????](https://arxiv.org/abs/2506.18248))


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
@inproceedings{jeong2026scga,
                title={Improving Black-Box Generative Attacks via Generator Semantic Consistency},
                author={Jeong, Jongoh and Yang, Hunmin and Jeong, Jaeseok and Yoon, Kuk-Jin},
                booktitle={International Conference on Learning Representations (ICLR)},
                year={2026}
                url={https://openreview.net/forum?id=ibXhUapwcz}
              }
```
