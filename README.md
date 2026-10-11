# J3 Reproducibility Package

**Spatio-temporal Characterisation of Diurnal Renewable Generation Patterns using Functional Principal Component Analysis**

This folder reproduces every manuscript figure and numerical result from a single notebook (`main.ipynb`) that runs top-to-bottom with only `DATA_DIR` and `RSA_SHP_PATH` set by the user.

---

## Repository layout

```
final/
├── main.ipynb            # master notebook — run this
├── config.py             # constants, TOU definitions, matplotlib style
├── data.py               # data loading and reshaping
├── fpca.py               # SVD, R², Tucker congruence, regressions
├── fpca_plots.py         # Figs. 2, 4–13
├── clustering.py         # K-means, stability, silhouette
├── clustering_plots.py   # Figs. 14–16, 18–19
├── figures/              # all saved PNGs (created on first run)
├── requirements.txt
└── README.md
```

---

## Data

Raw data are WASA 3×3 km normalised wind-power capacity factors for the Western Cape, South Africa.

- **DOI**: [10.5281/zenodo.20487179](https://doi.org/10.5281/zenodo.20487179)
- Files needed: `sam_power_wc_3x3_<year>.npz` for years 2014, 2015, 2016, 2017, 2019, plus `WC_3x3.csv` (location coordinates).
- A Western Cape boundary shapefile (`WC.shp`) is required for the spatial map figures.

---

## Installation

```bash
pip install -r requirements.txt
```

`geopandas` and `shapely` are required for the spatial map figures.  Install them via conda if pip fails on Windows:

```bash
conda install geopandas shapely
```

---

## Running

1. Open `main.ipynb` in Jupyter.
2. In the **Setup** cell (Section 0), set:
   ```python
   DATA_DIR     = Path("/path/to/wasa/data")
   RSA_SHP_PATH = "/path/to/WC.shp"
   ```
3. Run all cells (Kernel → Restart & Run All).

All figures are saved to `figures/` and displayed inline. After Section 6 completes, `KEY_RESULTS.md` is written to `final/` with the numerical results cited in the manuscript.

---

## Configuration

Global constants (years, Q, K, TOU definitions, K-means settings) live in `config.py`.  
Set `RUN_STABILITY_STUDY = True` to recompute the ARI stability study (≈200 × 5 K-means fits); the default `False` loads from cache if available.

---

## Figures produced

| Figure | Description |
|--------|-------------|
| Fig. 2  | Mean capacity factor map |
| Fig. 4  | Scree plot (R² about the mean daily profile) |
| Fig. 5  | PC loading vectors |
| Fig. 6  | Mean ± SD(score) × PC_q perturbations |
| Fig. 7  | Per-year vs pooled PC loadings (Tucker congruence) |
| Fig. 8  | Residual std map (Q=4 reconstruction) |
| Fig. 9  | Mean score maps PC1–PC4 |
| Fig. 10 | Seasonal (DJF / JJA) score maps |
| Fig. 11a| Evening peak vs CF (colour = PC1) |
| Fig. 11b| Evening peak vs CF (colour = PC2) |
| Fig. 12 | Morning mean vs CF (colour = PC2) |
| Fig. 13 | Midday mean vs CF (colour = PC3) |
| Fig. 14a| Elbow / ASW — FPCA features |
| Fig. 14b| Elbow / ASW — CF feature |
| Fig. 15 | Mean daily profiles by cluster (FPCA) |
| Fig. 16 | Silhouette — FPCA clustering |
| Fig. 18 | Mean daily profiles by cluster (CF) |
| Fig. 19 | Silhouette — CF clustering |

Figs. 3, 17, 20 are produced externally (QGIS / other scripts) and are not reproduced here.
