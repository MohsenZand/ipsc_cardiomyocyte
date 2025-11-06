#!/usr/bin/env python3
"""
Step 03: Comprehensive Figure Generation
Generates and assembles all panels for the main figures
(both single-cell and metacell versions).
"""
import sys
import os
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src import plotting as pl
from src import transcendental as tran

sc.settings.verbosity = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8

def plot_image_panel(ax, file_path, label_char):
    """Helper to plot a pre-generated image onto an axis."""
    if file_path and file_path.exists():
        img = Image.open(file_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f'Image not found\n{label_char}', ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    ax.set_title(label_char, fontsize=14, fontweight='bold', loc='left', pad=5)

def create_comprehensive_figure(adata_org, adata_seacell, transition_files, save_path, cell_level_heatmaps=False):
    """
    Main function to assemble the multi-panel figure.
    """
    print(f"Creating comprehensive figure: {save_path.name}")
    fig = plt.figure(figsize=(30, 12))
    gs = gridspec.GridSpec(3, 12, figure=fig, hspace=0.35, wspace=0.4,
                          left=0.04, right=0.98, top=0.96, bottom=0.04,
                          height_ratios=[1, 1.2, 1])

    # --- ROW 1: Cell-level UMAPs and Violins (always from adata_org) ---
    ax_a = fig.add_subplot(gs[0, 0:2]);  pl.plot_umap(adata_org, ax_a, 'entropy', 'A')
    ax_b = fig.add_subplot(gs[0, 2:4]);  pl.plot_umap(adata_org, ax_b, 'land', 'B')
    ax_c = fig.add_subplot(gs[0, 4:6]);  pl.plot_violin(adata_org, ax_c, 'entropy', 'C')
    ax_d = fig.add_subplot(gs[0, 6:8]);  pl.plot_violin(adata_org, ax_d, 'land', 'D')
    ax_e = fig.add_subplot(gs[0, 8:12]); pl.plot_flow_matrix(adata_org, ax_e, 'E')

    # --- ROW 2: Transition Plots (from adata_seacell) ---
    ax_f = fig.add_subplot(gs[1, 0:4]);  plot_image_panel(ax_f, transition_files.get('mpft'), 'F')
    ax_g = fig.add_subplot(gs[1, 4:8]);  plot_image_panel(ax_g, transition_files.get('3to4'), 'G')
    ax_h = fig.add_subplot(gs[1, 8:12]); plot_image_panel(ax_h, transition_files.get('10to4'), 'H')

    # --- ROW 3: Transcendental Heatmaps ---
    ax_i = fig.add_subplot(gs[2, 0:6])
    ax_j = fig.add_subplot(gs[2, 6:12])
    
    if cell_level_heatmaps:
        # Map memberships if they aren't already
        if 'rho_class' not in adata_org.obsm:
            adata_org = tran.map_seacell_memberships_to_cells(adata_org, adata_seacell)
        
        pl.plot_transcendental_heatmap(
            adata_org, '3', '4', ax_i, max_cells=5000, 
            region_method='adaptive', title='I: Transition A3 → A4 (Cells)'
        )
        pl.plot_transcendental_heatmap(
            adata_org, '10', '4', ax_j, max_cells=5000, 
            region_method='adaptive', title='J: Transition A10 → A4 (Cells)'
        )
    else: # Metacell-level heatmaps
        pl.plot_transcendental_heatmap(
            adata_seacell, '3', '4', ax_i, 
            region_method='logistic', title='I: Transition A3 → A4 (Metacells)'
        )
        pl.plot_transcendental_heatmap(
            adata_seacell, '10', '4', ax_j, 
            region_method='logistic', title='J: Transition A10 → A4 (Metacells)'
        )
    
    # --- Save Figure ---
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ...Saved {save_path.name}")

def main():
    print("\n" + "="*80)
    print("Generating All Final Figures")
    print("="*80)
    
    # 1. Load data
    print("Loading data...")
    adata_org = sc.read(config.MUTRANS_AD_ORG)
    adata_seacell = sc.read(config.MUTRANS_AD_SEACELL)

    # 2. Generate transition images (shared by both figures)
    print("Generating transition plots...")
    transition_plot_dir = config.FIGURES_DIR / '04_comprehensive_figures' / 'transition_plots'
    transition_files = pl.generate_mutrans_transition_plots(adata_seacell, transition_plot_dir)

    # 3. Create Metacell (SEACell) version of the figure
    save_path_seacells = config.FIGURES_DIR / '04_comprehensive_figures' / 'Fig_Comprehensive_Metacells.pdf'
    create_comprehensive_figure(
        adata_org, adata_seacell, transition_files, 
        save_path_seacells, cell_level_heatmaps=False
    )
    
    # 4. Create Single-Cell version of the figure
    save_path_cells = config.FIGURES_DIR / '04_comprehensive_figures' / 'Fig_Comprehensive_Cells.pdf'
    create_comprehensive_figure(
        adata_org, adata_seacell, transition_files, 
        save_path_cells, cell_level_heatmaps=True
    )
    
    print("\n" + "="*80)
    print("All figures generated!")
    print("="*80)

if __name__ == "__main__":
    main()