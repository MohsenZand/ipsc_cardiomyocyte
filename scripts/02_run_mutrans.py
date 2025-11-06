#!/usr/bin/env python3
"""
Step 02: Run MuTrans Analysis
Loads SEACell summaries and computes dynamical metrics.
Saves .h5ad files with MuTrans results and network tables.
"""
import sys
import scanpy as sc
import pandas as pd
import numpy as np
import collections.abc

# Add project root to path to allow 'from src import ...'
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # Import paths and params
from MuTrans.Example import pyMuTrans as pm

# --- Helper Functions (specific to this script) ---

def find_and_convert_matlab_objects(obj, key_path=None):
    """(Your function, unchanged)"""
    if key_path is None: key_path = []
    if isinstance(obj, collections.abc.Mapping):
        keys_to_convert = []
        for k, v in obj.items():
            current_path = key_path + [k]
            if isinstance(v, collections.abc.Mapping):
                find_and_convert_matlab_objects(v, key_path=current_path)
            elif 'matlab' in str(type(v)):
                keys_to_convert.append(k)
        for k in keys_to_convert:
            v = obj[k]
            print(f"Converting '{k}' at {' -> '.join(key_path + [k])} to numpy array.")
            try: obj[k] = np.asarray(v)
            except Exception: obj[k] = str(v)

def load_data():
    print("\n[1/4] Loading data files...")
    adata_org_seacell = sc.read(config.SEACELL_AD_ORG)
    adata_seacell = sc.read(config.SEACELL_AD_SUMMARY)
    print(f"  - SEACells summary: {adata_seacell.shape}")
    print(f"  - Original with SEACells: {adata_org_seacell.shape}")
    return adata_org_seacell, adata_seacell

def preprocess_and_cluster(adata_seacell):
    print("\n[2/4] Preprocessing and clustering...")
    # (Your QC, filtering, and clustering code from 'preprocess_seacells' 
    # and 'normalize_and_cluster' functions goes here)
    # ...
    sc.pp.filter_cells(adata_seacell, min_genes=300)
    sc.pp.filter_genes(adata_seacell, min_cells=10)
    # ... (all steps up to sc.tl.leiden) ...
    sc.tl.leiden(adata_seacell, resolution=config.LEIDEN_RESOLUTION, 
                key_added=f'leiden_{config.LEIDEN_RESOLUTION}')
    return adata_seacell

def run_mutrans(adata_seacell):
    print("\n[3/4] Running MuTrans dynamical analysis...")
    par = {
        "choice_distance": "cosine",
        "perplex": config.MUTRAMS_PERPLEX,
        "K_cluster": config.MUTRAMS_K_CLUSTER,
        "reduction_coord": 'umap',
        # (Rest of your 'par' dict)
        "force_double_precision": True,
    }
    adata_seacell = pm.dynamical_analysis(adata_seacell, par)
    return adata_seacell

def transfer_and_save(adata_seacell, adata_org_seacell):
    print("\n[4/4] Transferring metrics and saving...")
    # Transfer metrics
    ent_map = adata_seacell.obs['entropy'].to_dict()
    adata_org_seacell.obs['entropy'] = adata_org_seacell.obs['SEACell'].map(ent_map)
    for key in ['land', 'attractor']:
        if key in adata_seacell.obs:
            m = adata_seacell.obs[key].to_dict()
            adata_org_seacell.obs[key] = adata_org_seacell.obs['SEACell'].map(m)

    # Convert and save
    find_and_convert_matlab_objects(adata_org_seacell.uns)
    adata_org_seacell.write(config.MUTRANS_AD_ORG)
    print(f"  Saved: {config.MUTRANS_AD_ORG.name}")
    
    find_and_convert_matlab_objects(adata_seacell.uns)
    adata_seacell.write(config.MUTRANS_AD_SEACELL)
    print(f"  Saved: {config.MUTRANS_AD_SEACELL.name}")
    
    # --- Network Table Export ---
    # (Your 'analyze_lineage_network' logic for *table generation* goes here)
    # This part should NOT plot, just save the CSVs.
    transition_matrix = adata_seacell.uns['land']['land']
    # ... (your logic to build 'network_df' and 'attractor_network_df')
    # network_df.to_csv(config.TABLES_DIR / 'land_lineage_network_table.csv', index=False)
    # attractor_network_df.to_csv(config.TABLES_DIR / 'attractor_lineage_network_aggregated.csv', index=False)
    # print("  Saved network tables.")

def main():
    print("=" * 80)
    print("MuTrans Analysis Pipeline")
    print("=" * 80)
    adata_org_seacell, adata_seacell = load_data()
    adata_seacell = preprocess_and_cluster(adata_seacell)
    adata_seacell = run_mutrans(adata_seacell)
    transfer_and_save(adata_seacell, adata_org_seacell)
    print("\n" + "=" * 80)
    print("Analysis complete! Processed .h5ad files saved.")
    print("=" * 80)

if __name__ == "__main__":
    main()