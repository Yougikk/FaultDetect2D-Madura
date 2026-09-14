# FaultDetect2D-Madura

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ITS](https://img.shields.io/badge/ITS-Teknik%20Geofisika-003f87.svg)](https://www.its.ac.id/)

Automated 2D seismic fault detection pipeline for **vintage Madura RMKS Fault Zone data**, using a **U-Net 2D** trained from scratch with the [FaultSSL](https://doi.org/10.1190/geo2023-0550.1) supervised baseline augmentation strategy on [FaultSeg3D](https://doi.org/10.1190/geo2018-0646.1) synthetic data. Pre-inference conditioning combines a zero-phase Butterworth bandpass filter and an anisotropic Gaussian **Structure-Oriented Filter (SOF)** to suppress random noise while preserving fault discontinuity sharpness.

---

## Pipeline Overview

```
Vintage 2D SEG-Y  (1980s–1990s, Madura RMKS Fault Zone)
         │
         ▼
┌─────────────────────────────────────────────┐
│  Step 1 · Seismic Conditioning              │
│  Butterworth bandpass  (10–30 Hz, order 4)  │
│  → SOF anisotropic Gaussian                 │
│    (σ_dip = 2.5 · σ_norm = 0.5)            │
│  → Z-score normalization per trace          │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Step 2 · U-Net 2D Training                 │
│  FaultSeg3D synthetic data                  │
│  + FaultSSL supervised baseline aug.        │
│    Gaussian noise · Gamma · Gaussian blur   │
│  Dice Loss · AdamW · ReduceLROnPlateau      │
│  Early stopping (patience 20)               │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Step 3 · Inference + Post-processing       │
│  Sliding window  (128×128 px, stride 64)    │
│  Hann-window blending                       │
│  → Hysteresis threshold  (P95-adaptive)     │
│  → Morphological closing (3×3 px)           │
│  → Remove small objects  (< 20 px)          │
│  → Skeletonization (medial axis)            │
│  → Geological constraint filter             │
│    (length ≥ 20 px · eccentricity ≥ 0.90)  │
└─────────────────────────────────────────────┘
         │
         ▼
Fault picks — skeleton overlay on seismic section
```

---

## Results

**Model performance** on the FaultSeg3D synthetic validation set (best epoch 44 of 45):

| Metric        | Value  |
|---------------|--------|
| Dice Loss     | 0.2530 |
| IOU           | 0.6052 |
| Precision     | 0.7705 |
| Recall        | 0.7770 |
| F1-Score      | 0.7470 |

**Applied to three Madura 2D seismic lines** (selected by SNR + spatial coherence QC):

| Line      | Quality       | SNR (dB) | Segments | Mean Apparent Dip | % Dip > 60° |
|-----------|---------------|----------|----------|-------------------|-------------|
| 83-MDR-1  | Good          | 2.45     | 80       | 76.6°             | 98.8%       |
| 83-MDR-3  | Fair          | 2.59     | 127      | 76.4°             | 95.3%       |
| 83-MDR-11 | Poor          | 1.50     | 163      | 76.8°             | 98.2%       |

Dominant sub-vertical to vertical fault geometry (mean dip 76.4°–76.8°, consistent across all quality levels) is consistent with the RMKS wrench-fault system and positive flower structure documented by Satyana et al. (2004).

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Tested on Python 3.10, PyTorch 2.0, CUDA 11.8 (NVIDIA RTX 3050 4 GB). CPU inference is supported but slower.

### 1. Prepare FaultSeg3D training data

Download the synthetic training volumes from the [FaultSeg3D Google Drive](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRLP8) and arrange them as:

```
data/
└── faultseg3d/
    ├── train/
    │   ├── seis/     # 200 × (128×128×128) float32 seismic volumes
    │   └── fault/    # 200 × (128×128×128) binary fault labels
    └── val/
        ├── seis/     # 20 × (128×128×128) float32 seismic volumes
        └── fault/    # 20 × (128×128×128) binary fault labels
```

### 2. Run the three notebooks in order

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [`notebooks/01_seismic_preprocessing.ipynb`](notebooks/01_seismic_preprocessing.ipynb) | Load SEG-Y → QC (SNR, coherence) → Bandpass → SOF → Z-score → save `.npy` |
| 2 | [`notebooks/02_faultssl_training.ipynb`](notebooks/02_faultssl_training.ipynb)         | Extract 2D slices from FaultSeg3D → FaultSSL augmentation → train U-Net 2D |
| 3 | [`notebooks/03_inference_pipeline.ipynb`](notebooks/03_inference_pipeline.ipynb)       | Load `.npy` → sliding-window inference → post-processing → fault picks |

### 3. Configure paths

All paths and parameters are set in the `CONFIG` dict at **Cell 2** of each notebook. Replace the example paths with your local directories before running.

> **Key paths to update:**
> - `input_dir` / `output_dir` in notebook 1
> - `dir_seis`, `dir_fault`, `output_dir` in notebook 2
> - `model_path_2d`, `npy_dir`, `output_dir` in notebook 3

---

## Pretrained Model

The best U-Net 2D checkpoint (epoch 44, val Dice Loss 0.2530) is available for download:

> 📥 **[Download `faultssl_best_model.pth` — Google Drive](#)** *(link will be added)*

Place it at `checkpoints/faultssl_best_model.pth` and set `model_path_2d` in notebook 3 accordingly.

---

## Repository Structure

```
FaultDetect2D-Madura/
│
├── notebooks/
│   ├── 01_seismic_preprocessing.ipynb    # Bandpass + SOF + QC
│   ├── 02_faultssl_training.ipynb        # U-Net training with FaultSSL augmentation
│   └── 03_inference_pipeline.ipynb       # Sliding-window inference + post-processing
│
├── results/                              # Output figures (PNG)
│   ├── preprocessing_comparison.png
│   ├── training_curves.png
│   ├── fault_probability_maps.png
│   └── final_fault_picks.png
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Key Design Decisions

**Why U-Net 2D instead of FaultNet 3D?**
The Madura vintage data is 2D post-stack. Slicing each FaultSeg3D cube along the crossline axis produces 108 patches per cube (indices 10–118, excluding boundary effects), yielding 21,600 training and 2,160 validation 128×128 patches. The 2D formulation keeps GPU memory requirements low (runs on a 4 GB GPU with batch size 8) while being directly applicable to 2D seismic sections.

**Why the FaultSSL augmentation strategy?**
Standard FaultSeg3D training produces a model optimized for clean synthetic data. Vintage Madura data (1980s–1990s) has significantly lower SNR. The FaultSSL supervised baseline augmentation — Gaussian noise `σ ~ U(0, 0.15)`, gamma transform, and Gaussian blur — artificially widens the training distribution to bridge the domain gap, following Dou et al. (2024) Section II.E.

**Why P95-adaptive hysteresis thresholding?**
Vintage lines of different quality produce probability maps with different dynamic ranges. A fixed threshold applied uniformly would systematically under-detect on the poor-quality line (83-MDR-11, mean prob. 0.0944) while over-detecting on the good-quality line (83-MDR-1, mean prob. 0.1294). Per-line P95 calibration ensures a fair, data-driven threshold for each section.

---

## Citation

If you use this code or data in your research, please cite:

**This work:**
```bibtex
@thesis{wahyudi2026faultdetect,
  author = {Yogi Ananta Wahyudi},
  title  = {Otomatisasi Deteksi Sesar pada Data Seismik 2D {Vintage} {Madura}
            Menggunakan {Supervised Baseline FaultSSL} dan
            {Structure-Oriented Filter}},
  school = {Institut Teknologi Sepuluh Nopember},
  year   = {2026},
  type   = {Undergraduate Thesis},
  url    = {https://github.com/[your-username]/FaultDetect2D-Madura}
}
```

**FaultSeg3D** (training data & U-Net architecture):
```bibtex
@article{wu2019faultSeg,
  author  = {Xinming Wu and Luming Liang and Yunzhi Shi and Sergey Fomel},
  title   = {{FaultSeg3D}: using synthetic datasets to train an end-to-end
             convolutional neural network for {3D} seismic fault segmentation},
  journal = {Geophysics},
  volume  = {84},
  number  = {3},
  pages   = {IM35--IM45},
  year    = {2019},
  doi     = {10.1190/geo2018-0646.1}
}
```

**FaultSSL** (augmentation strategy):
```bibtex
@article{dou2024faultssl,
  author  = {Yuxing Dou and Kewen Li and Min Dong and Yue Xiao},
  title   = {{FaultSSL}: seismic fault detection via semi-supervised learning},
  journal = {Geophysics},
  volume  = {89},
  number  = {3},
  pages   = {M79--M91},
  year    = {2024},
  doi     = {10.1190/geo2023-0550.1}
}
```

**Structure-Oriented Filter:**
```bibtex
@article{fehmers2003sof,
  author  = {Gijs C. Fehmers and Christian F. W. H\"{o}cker},
  title   = {Fast structural interpretation with structure-oriented filtering},
  journal = {Geophysics},
  volume  = {68},
  number  = {4},
  pages   = {1286--1293},
  year    = {2003},
  doi     = {10.1190/1.1598121}
}
```

---

## License

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) —
consistent with the [FaultSeg3D repository](https://github.com/xinwucwp/faultSeg).
For commercial use, please contact the authors.

---

## Acknowledgements

- **Wu et al. (2019)** — FaultSeg3D synthetic dataset and U-Net baseline
- **Dou et al. (2024)** — FaultSSL supervised baseline augmentation strategy
- **Fehmers & Höcker (2003); Hale (2009)** — Structure-Oriented Filter
- **Pusat Survei Geologi Bandung** — vintage 2D seismic data from Madura
- **Advisors:** Dr. Ir. Firman Syaifuddin, S.Si, M.T. and Ir. Mu'lif Luthfy Sholih, S.T., M.Eng. — Departemen Teknik Geofisika, ITS
