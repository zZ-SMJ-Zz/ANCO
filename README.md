# Out-of-Distribution Detection with Anisotropic Cosine Similarity


## Overview

While cosine similarity is widely used in OOD detection, its isotropic assumption leads to suboptimal detection boundaries in anisotropic embedding spaces. We propose **ANCO** (Anisotropic Cosine similarity), which rescales angular deviations based on direction-dependent in-distribution dispersion estimated via the Fréchet mean and tangent-space covariance on the unit hypersphere.

We also introduce **ANG**, which applies ANCO to [NNGuide](https://github.com/roomo7time/nnguide) using 1,000 class means.

### Key idea

Given a test feature $\mathbf{x}$ and class prototype $\boldsymbol{\mu}_c$ (Fréchet mean on $\mathbb{S}^{D-1}$), ANCO computes:

```
θ'_ci = min(θ_ci · (σ̄ / σ(u_ci)), π)
s_c(x) = cos(θ'_ci)
S(x)   = max_c s_c(x)
```

where `σ(u_ci)` is the direction-dependent standard deviation from the tangent-space covariance, and `σ̄` is the global average standard deviation.

## This repository

This codebase extends [Mahalanobis++](https://github.com/MMaximilianMueller/mahalanobis-plus-plus) (Müller & Hein, ICML 2025) with ANCO and ANG. The evaluation framework, data loading, and baselines are inherited from that repository, which in turn builds on [NINCO](https://github.com/j-cb/NINCO) and [OpenOOD](https://github.com/Jingkang50/OpenOOD).

Key additions:
- `NINCO/detection_methods.py`: `evaluate_ANCO()`, `evaluate_ANCOGuide()`
- `OpenOOD/eval_cifar.py`: ANCO and ANG evaluation for CIFAR-100 benchmarks

## Setup

```bash
conda env create -f anco.yml
conda activate anco
pip install libmr==0.1.9
```

### Download data

```bash
# NINCO datasets
cd NINCO
wget https://zenodo.org/record/8013288/files/NINCO_all.tar.gz?download=1 -O NINCO_all.tar.gz
tar -xvzf NINCO_all.tar.gz

# OpenOOD datasets and checkpoints
cd ../OpenOOD
python ./scripts/download/download.py --contents 'datasets' 'checkpoints' --datasets 'ood_v1.5' --checkpoints 'ood_v1.5' --save_dir './data' './results' --dataset_mode 'benchmark'
```

Then edit `NINCO/data/paths_config.py` to set:
- `ninco_folder`: path to the extracted NINCO datasets (containing `NINCO_OOD_classes`, `NINCO_OOD_unit_tests`, `NINCO_popular_datasets_subsamples`)
- `repo_path`: absolute path to this repository
- ImageNet path

## Running ANCO and ANG

### ImageNet (NINCO / OpenOOD datasets)

Run:

```bash
# On OpenOOD datasets (NINCO, SSB-hard, Texture, OpenImages-O, iNaturalist)
python evaluate.py --model_name eva02_large_patch14_448.mim_m38m_ft_in1k --method anco --dataset openood-datasets --dataset_paths_prefix ./OpenOOD/data/

# ANG (ANCO applied to NNGuide)
python evaluate.py --model_name eva02_large_patch14_448.mim_m38m_ft_in1k --method ancoguide --dataset openood-datasets --dataset_paths_prefix ./OpenOOD/data/
```

> Note: The first run requires a full forward pass over ImageNet-1K train set to extract features. Extracted features are cached for subsequent runs.

### CIFAR-100

Run from the `OpenOOD` folder:

```bash
python eval_cifar.py
```

## Results

### ImageNet-1K (EVA02-L14, avg FPR ↓ across 5 OOD datasets)

| Method | Avg FPR ↓ | Avg AUC ↑ |
|--------|-----------|-----------|
| Cosine | 20.6 | 95.4 |
| NNGuide | 25.9 | 91.4 |
| Mahalanobis++ | 17.7 | 96.0 |
| **ANCO (ours)** | **17.3** | **96.0** |
| **ANG (ours)** | **15.9** | **96.2** |

### CIFAR-100 (avg FPR ↓ across 4 ResNet backbones)

| Method | Avg FPR ↓ |
|--------|-----------|
| Cosine | 69.88 |
| NNGuide | 72.42 |
| Mahalanobis++ | 69.72 |
| **ANCO (ours)** | **67.35** |
| **ANG (ours)** | **66.35** |

## Citation

If you use this code, please also cite the repositories this work builds upon:

```bibtex
@inproceedings{mueller2025mahalanobispp,
  title={Mahalanobis++: Improving OOD Detection via Feature Normalization},
  author={Maximilian Mueller and Matthias Hein},
  booktitle={ICML},
  year={2025},
  url={https://arxiv.org/abs/2505.18032}
}

@inproceedings{bitterwolf2023ninco,
  title={In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation},
  author={Julian Bitterwolf and Maximilian Mueller and Matthias Hein},
  booktitle={ICML},
  year={2023},
  url={https://proceedings.mlr.press/v202/bitterwolf23a.html}
}

@article{zhang2023openood,
  title={OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection},
  author={Zhang, Jingyang and Yang, Jingkang and Wang, Pengyun and Wang, Haoqi and Lin, Yueqian and Zhang, Haoran and Sun, Yiyou and Du, Xuefeng and Li, Yixuan and Liu, Ziwei and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2306.09301},
  year={2023}
}

@inproceedings{park2023nearest,
  title={Nearest Neighbor Guidance for Out-of-Distribution Detection},
  author={Park, Jaewoo and Jung, Dong Gyu and Kim, Hwanjun and Han, Sanghyuk},
  booktitle={ICCV},
  year={2023}
}
```
