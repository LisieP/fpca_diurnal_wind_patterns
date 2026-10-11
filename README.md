# FPCA for characterising diurnal renewable energy generation patterns

This code accompanies the publication:
**Spatio-temporal Characterisation of Diurnal Renewable Generation Patterns using Functional Principal Component Analysis**, "<DOI>"

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

