from pathlib import Path

# --- Project Root ---
# Assumes this file is in 'ipsc_cardiomyocyte_dynamics/src/'
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Parameters ---
LEIDEN_RESOLUTION = 0.5
N_TOP_HVG = 3000
SEACELL_N_WAYPOINT_EIGS = 10
SEACELL_CONVERGENCE_EPS = 1e-5
MUTRAMS_PERPLEX = 200.0
MUTRAMS_K_CLUSTER = 14

# --- Input Data ---
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
RAW_H5AD = RAW_DATA_DIR / 'GSE175634_iPSC_CM.sct3k_reclustered.h5ad'

# --- Processed Data Paths ---
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
SEACELL_DIR = PROCESSED_DATA_DIR / 'seacells'
SEACELL_AD_ORG = SEACELL_DIR / 'output_with_SEACells.h5ad'
SEACELL_AD_SUMMARY = SEACELL_DIR / 'SEACell_summary.h5ad'
SEACELL_AD_SOFT = SEACELL_DIR / 'SEACell_soft_summary.h5ad'

MUTRANS_DIR = PROCESSED_DATA_DIR / 'mutrans'
MUTRANS_AD_SEACELL = MUTRANS_DIR / f'seacells_mutrans_{LEIDEN_RESOLUTION}.h5ad'
MUTRANS_AD_ORG = MUTRANS_DIR / f'seacells_org_mutrans_{LEIDEN_RESOLUTION}.h5ad'

# --- Results ---
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

# Ensure directories exist
DIRS_TO_MAKE = [
    SEACELL_DIR, MUTRANS_DIR, FIGURES_DIR, TABLES_DIR,
    FIGURES_DIR / '01_seacell_qc',
    FIGURES_DIR / '02_mutrans_analysis',
    FIGURES_DIR / '03_transcendental_heatmaps',
    FIGURES_DIR / '04_comprehensive_figures'
]
for d in DIRS_TO_MAKE:
    d.mkdir(parents=True, exist_ok=True)