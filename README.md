# Improving Black-Box Generative Attacks via Generator Semantic Consistency (ICLR 2026)

This official repository contains the implementation of SCGA, a semantic structure-aware add-on method to improve black-box adversarial transferability of generative attacks.

🌐 **Project Site**: [https://andyj1.github.io/SCGA/](https://andyj1.github.io/SCGA/)

📖 **Paper**: [arXiv:2506.18248](https://arxiv.org/pdf/2506.18248)

**Jongoh Jeong**<sup>1</sup>, **Hunmin Yang**<sup>1,2</sup>, **Jaeseok Jeong**<sup>1</sup>, **Kuk-Jin Yoon**<sup>1</sup>

<sup>1</sup>KAIST  
<sup>2</sup>ADD


## 📄 Abstract

Transfer attacks optimize on a surrogate and deploy to a black-box target. While iterative optimization attacks in this paradigm are limited by their per-input cost limits efficiency and scalability due to multistep gradient updates for each input, generative attacks alleviate these by producing adversarial examples in a single forward pass at test time. However, current generative attacks still adhere to optimizing surrogate losses (e.g., feature divergence) and overlook the generator's internal dynamics, underexploring how the generator's internal representations shape transferable perturbations. To address this, we enforce semantic consistency by aligning the early generator's intermediate features to an EMA teacher, stabilizing object-aligned representations and improving black-box transfer without inference-time overhead. To ground the mechanism, we quantify semantic stability as the standard deviation of foreground IoU between cluster-derived activation masks and foreground masks across generator blocks, and observe reduced semantic drift under our method. For more reliable evaluation, we also introduce Accidental Correction Rate (ACR) to separate inadvertent corrections from intended misclassifications, complementing the inherent blind spots in traditional Attack Success Rate (ASR), Fooling Rate (FR), and Accuracy metrics. Across architectures, domains, and tasks, our approach can be seamlessly integrated into existing generative attacks with consistent improvements in black-box transfer, while maintaining test-time efficiency.
## 🔧 Requirements

- **Python**: >= 3.10
- **PyTorch**: 2.6.0 (with CUDA 11.8)
- **CUDA**: 11.8+

## 📦 Installation

### Clone the Repository
```bash
git clone https://github.com/andyj1/SCGA.git
cd SCGA
conda create -f environment.yaml
```

## 📊 Datasets

This repository supports the following datasets for training and evaluation:

### ImageNet
- **Full Name**: ImageNet-1K
- **Structure**: Standard ImageFolder format
  ```
  imagenet/
  ├── train/
  │   ├── class1/
  │   │   ├── img1.jpg
  │   │   └── img2.jpg
  │   └── class2/
  └── val/
      ├── class1/
      └── class2/
  ```
- **Usage**: Set `--train_data imagenet` in training script

### CUB-200-2011
- **Full Name**: Caltech-UCSD Birds-200-2011
- **Structure**: Requires metadata files in the dataset root
  ```
  cub/
  ├── images/
  │   ├── 001.Black_footed_Albatross/
  │   └── 002.Laysan_Albatross/
  ├── images.txt
  ├── image_class_labels.txt
  └── train_test_split.txt
  ```
- **Usage**: Set `--train_data cub` in training script

### Stanford Cars (CAR)
- **Full Name**: Stanford Cars Dataset
- **Structure**: Requires annotation files and images
  ```
  car/
  ├── cars_train/
  │   └── [images]
  ├── cars_test/
  │   └── [images]
  ├── devkit/
  │   ├── cars_train_annos.mat
  │   └── cars_meta.mat
  ├── train.txt
  └── test.txt
  ```
- **Usage**: Set `--train_data car` in training script

### FGVC Aircraft (AIR)
- **Full Name**: FGVC-Aircraft Dataset
- **Structure**: Standard FGVC-Aircraft format
  ```
  air/
  ├── fgvc-aircraft-2013b/
  │   ├── data/
  │   │   ├── images/
  │   │   │   ├── [aircraft images]
  │   │   ├── images_variant_train.txt
  │   │   ├── images_variant_test.txt
  │   │   └── variants.txt
  ├── train.txt
  └── test.txt
  ```
- **Usage**: Set `--train_data air` in training script

### Data Directory Setup

Set the `DATA_ROOT_DIR` in `datasets/paths.py` to point to your datasets directory. The expected structure is:

```
DATA_ROOT_DIR/
├── imagenet/
├── cub/
├── car/
└── air/
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
