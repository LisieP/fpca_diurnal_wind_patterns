"""
Central configuration for the J3 FPCA manuscript reproducibility package.
Set DATA_DIR to the folder containing the .npz files before running main.ipynb.
"""

from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
# The user sets DATA_DIR in the Setup cell of main.ipynb.
# OUTPUT_DIR receives all figures and cached results.
DATA_DIR: Path = Path("SET_ME")          # override in notebook Setup cell
OUTPUT_DIR: Path = Path("figures")       # relative to final/; override if needed

# ── dataset ──────────────────────────────────────────────────────────────────
YEARS = [2014, 2015, 2016, 2017, 2019]   # 2018 absent from dataset
H = 24                                    # hourly values per day
D = 1825                                  # total days after dropping 29 Feb 2016

# ── FPCA ─────────────────────────────────────────────────────────────────────
Q = 4                                     # retained modes

# ── K-means ──────────────────────────────────────────────────────────────────
K = 7                                     # number of clusters
KMEANS_N_INIT = 50                        # number of random restarts for final run
KMEANS_SEED = 42

# ── stability study ───────────────────────────────────────────────────────────
RUN_STABILITY_STUDY = False               # set True to recompute; False loads cache
STABILITY_N_RUNS = 200
STABILITY_N_INIT_VALUES = (1, 5, 10, 20, 50)
STABILITY_SEED_OFFSET = 1000

# ── TOU / season definitions (matches cell 6 of original notebook) ───────────
HDS_MONTHS = [6, 7, 8]
LDS_MONTHS = [1, 2, 3, 4, 5, 9, 10, 11, 12]

HDS_TOU = {
    "off_peak": [1, 2, 3, 4, 5, 6, 23, 0],
    "standard": [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22],
    "m_peak":   [7, 8, 9],
    "e_peak":   [18, 19],       # HDS evening peak: 18:00–19:59
    "peak":     [7, 8, 9, 18, 19],
}
LDS_TOU = {
    "off_peak": [1, 2, 3, 4, 5, 6, 23, 0],
    "standard": [7, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
    "m_peak":   [8, 9, 10],
    "e_peak":   [19, 20],       # LDS evening peak: 19:00–20:59
    "peak":     [8, 9, 10, 19, 20],
}

MORNING_HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
MIDDAY_HOURS  = [10, 11, 12, 13, 14]

SUMMER_MONTHS = [12, 1, 2]   # DJF
WINTER_MONTHS = [6, 7, 8]    # JJA

# ── matplotlib style ─────────────────────────────────────────────────────────
import seaborn as sns
import matplotlib.pyplot as plt

_BRIGHT = sns.color_palette("bright")
_BRIGHT[2] = "#17A612"

COLORS = [
    _BRIGHT[0], _BRIGHT[2], _BRIGHT[3],
    _BRIGHT[1], _BRIGHT[4], _BRIGHT[9], _BRIGHT[5],
]

MARKERS = ["P", "^", "o", "D", "s", "p", "*", "h", "H", "D", "d", "P", "X"]


def apply_rcparams():
    """Apply the manuscript matplotlib style."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.titlesize"] = 28
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.figsize"] = (12, 7)
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_BRIGHT)
