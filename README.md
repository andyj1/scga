# Improving Black-Box Generative Attacks via Generator Semantic Consistency (ICLR 2026)

This official repository contains the implementation of SCGA, a semantic structure-aware add-on method to improve black-box adversarial transferability of generative attacks.

🌐 **Project Site**: [https://andyj1.github.io/SCGA/](https://andyj1.github.io/SCGA/)

📖 **Paper**: [arXiv:2506.18248](https://arxiv.org/pdf/2506.18248)

## 🔧 Requirements

- **Python**: >= 3.10
- **PyTorch**: 2.6.0 (with CUDA 11.8)
- **CUDA**: 11.8+

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/andyj1/SCGA.git
cd SCGA
conda create -f environment.yaml
```

## 📝 Training the Generator

To train the generator, run:

```bash
python train_BIA_ours.py
```

For more options and arguments, see the argument parser in `utee/parser.py`.


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
