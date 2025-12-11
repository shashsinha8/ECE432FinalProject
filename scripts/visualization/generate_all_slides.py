#!/usr/bin/env python3
"""
Generate All Presentation Slides

This script generates all visualization plots for presentation slides
with consistent styling and color palette.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Color palette
colors = {
    "classical": "#4A4A4A",      # dark gray
    "hard_ml": "#9E9E9E",        # light gray
    "soft_ml": "#2E7D32",        # dark green
    "deep": "#1B5E20",           # deep green
    "wide": "#66BB6A",           # mid green
    "residual": "#00C853",       # bright green
}

# Styling constants
GRID_COLOR = "#E0E0E0"
LINE_WIDTH = 2.2
MARKER_SIZE = 5
FIG_SIZE = (7, 5)


def load_results():
    """Load evaluation results."""
    results_path = project_root / "data" / "phase5_final_results.npy"
    return np.load(results_path, allow_pickle=True).item()


def setup_plot(figsize=FIG_SIZE):
    """Set up plot with consistent styling."""
    plt.figure(figsize=figsize, facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    return ax


def save_plot(filename, dpi=300):
    """Save plot with consistent settings."""
    plt.tight_layout()
    output_path = project_root / "data" / filename
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()


def slide6_classical_ber():
    """Slide 6: Classical Decoder BER Curve"""
    results = load_results()
    classical_data = results["results"]["classical"]
    ebno = classical_data["ebno_range"]
    classical_ber = classical_data["ber"]
    
    setup_plot()
    plt.plot(ebno, classical_ber, 
             marker="o", markersize=MARKER_SIZE,
             color=colors["classical"], linewidth=LINE_WIDTH,
             markerfacecolor=colors["classical"],
             markeredgecolor=colors["classical"])
    
    plt.yscale("log")
    plt.xlabel("Eb/N0 (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (log scale)", fontsize=12)
    plt.title("Classical Hamming(7,4) Decoder Performance", fontsize=14, fontweight='bold')
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6, color=GRID_COLOR)
    plt.xlim(left=ebno.min() - 0.5, right=ebno.max() + 0.5)
    plt.ylim(bottom=max(1e-5, classical_ber.min() * 0.5), top=min(1.0, classical_ber.max() * 2))
    
    save_plot("classical_ber.png")


def slide_comparison_classical_vs_soft():
    """Comparison: Classical vs Soft-Decision ML"""
    results = load_results()
    classical_data = results["results"]["classical"]
    soft_data = results["results"]["soft_ml"]
    
    ebno = classical_data["ebno_range"]
    classical_ber = classical_data["ber"]
    soft_ber = soft_data["ber"]
    
    setup_plot()
    plt.plot(ebno, classical_ber, 
             marker="o", markersize=MARKER_SIZE,
             color=colors["classical"], linewidth=LINE_WIDTH,
             label="Classical Decoder",
             markerfacecolor=colors["classical"],
             markeredgecolor=colors["classical"])
    
    plt.plot(ebno, soft_ber, 
             marker="o", markersize=MARKER_SIZE,
             color=colors["soft_ml"], linewidth=LINE_WIDTH,
             label="Soft-Decision ML",
             markerfacecolor=colors["soft_ml"],
             markeredgecolor=colors["soft_ml"])
    
    plt.yscale("log")
    plt.xlabel("Eb/N0 (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (log scale)", fontsize=12)
    plt.title("Classical vs Soft-Decision ML Decoder", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6, color=GRID_COLOR)
    plt.xlim(left=ebno.min() - 0.5, right=ebno.max() + 0.5)
    
    save_plot("comparison_classical_vs_soft.png")


def slide_all_decoders_comparison():
    """Comprehensive comparison of all decoders"""
    results = load_results()
    classical_data = results["results"]["classical"]
    ebno = classical_data["ebno_range"]
    
    setup_plot(figsize=(10, 6))
    
    # Plot order: classical, hard_ml, soft_ml, phase3, phase4 variants
    plot_order = ["classical", "hard_ml", "soft_ml", "phase3", 
                  "phase4_deep", "phase4_wide", "phase4_residual"]
    
    labels_map = {
        "classical": "Classical Decoder",
        "hard_ml": "Hard-Decision ML",
        "soft_ml": "Soft-Decision ML",
        "phase3": "Phase 3 (Improved Training)",
        "phase4_deep": "Deep Architecture",
        "phase4_wide": "Wide Architecture",
        "phase4_residual": "Residual Architecture"
    }
    
    for key in plot_order:
        if key in results["results"]:
            data = results["results"][key]
            color_key = key.replace("phase4_", "").replace("phase3", "soft_ml")
            if color_key not in colors:
                color_key = "soft_ml"  # fallback
            
            plt.plot(data["ebno_range"], data["ber"],
                    marker="o", markersize=MARKER_SIZE,
                    color=colors[color_key], linewidth=LINE_WIDTH,
                    label=labels_map.get(key, key),
                    markerfacecolor=colors[color_key],
                    markeredgecolor=colors[color_key])
    
    plt.yscale("log")
    plt.xlabel("Eb/N0 (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (log scale)", fontsize=12)
    plt.title("Comprehensive Decoder Performance Comparison", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, framealpha=0.9, loc='upper right')
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6, color=GRID_COLOR)
    plt.xlim(left=ebno.min() - 0.5, right=ebno.max() + 0.5)
    
    save_plot("comparison_all_decoders_presentation.png")


def slide_phase4_architectures():
    """Phase 4: Architecture Variants Comparison"""
    results = load_results()
    classical_data = results["results"]["classical"]
    ebno = classical_data["ebno_range"]
    
    setup_plot()
    
    # Plot classical as baseline
    plt.plot(ebno, classical_data["ber"],
            marker="o", markersize=MARKER_SIZE,
            color=colors["classical"], linewidth=LINE_WIDTH,
            label="Classical Decoder",
            markerfacecolor=colors["classical"],
            markeredgecolor=colors["classical"],
            linestyle="--", alpha=0.7)
    
    # Plot Phase 4 architectures
    phase4_models = {
        "phase4_deep": ("Deep Architecture", colors["deep"]),
        "phase4_wide": ("Wide Architecture", colors["wide"]),
        "phase4_residual": ("Residual Architecture", colors["residual"])
    }
    
    for key, (label, color) in phase4_models.items():
        if key in results["results"]:
            data = results["results"][key]
            plt.plot(data["ebno_range"], data["ber"],
                    marker="o", markersize=MARKER_SIZE,
                    color=color, linewidth=LINE_WIDTH,
                    label=label,
                    markerfacecolor=color,
                    markeredgecolor=color)
    
    plt.yscale("log")
    plt.xlabel("Eb/N0 (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (log scale)", fontsize=12)
    plt.title("Phase 4: Architecture Variants Comparison", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6, color=GRID_COLOR)
    plt.xlim(left=ebno.min() - 0.5, right=ebno.max() + 0.5)
    
    save_plot("phase4_architectures_comparison.png")


def main():
    """Generate all presentation slides."""
    print("=" * 70)
    print("Generating Presentation Visualizations")
    print("=" * 70)
    print()
    
    print("1. Slide 6: Classical Decoder BER Curve")
    slide6_classical_ber()
    print()
    
    print("2. Comparison: Classical vs Soft-Decision ML")
    slide_comparison_classical_vs_soft()
    print()
    
    print("3. Comprehensive: All Decoders Comparison")
    slide_all_decoders_comparison()
    print()
    
    print("4. Phase 4: Architecture Variants")
    slide_phase4_architectures()
    print()
    
    print("=" * 70)
    print("All visualizations generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

