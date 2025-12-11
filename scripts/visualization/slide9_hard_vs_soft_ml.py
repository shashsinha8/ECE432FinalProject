#!/usr/bin/env python3
"""
Slide 9: Hard ML vs Soft ML vs Classical

Generates a comparison visualization of Classical, Hard-Decision ML, and 
Soft-Decision ML decoder performance with presentation-ready styling.
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

# Load results
results_path = project_root / "data" / "phase5_final_results.npy"
results = np.load(results_path, allow_pickle=True).item()

# Extract data from nested structure
classical_data = results["results"]["classical"]
ebno = classical_data["ebno_range"]
ber_classical = classical_data["ber"]

# Extract hard ML data (if available)
if "hard_ml" in results["results"]:
    hard_ml_data = results["results"]["hard_ml"]
    ber_ml_hard = hard_ml_data["ber"]
else:
    print("Warning: Hard ML data not found in results")
    ber_ml_hard = None

# Extract soft ML data
soft_ml_data = results["results"]["soft_ml"]
ber_ml_soft = soft_ml_data["ber"]

# Create figure with white background
plt.figure(figsize=(7, 5), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Plot Classical decoder
plt.plot(ebno, ber_classical, 
         marker="o", 
         markersize=5,
         color=colors["classical"], 
         linewidth=2.2,
         label="Classical",
         markerfacecolor=colors["classical"],
         markeredgecolor=colors["classical"])

# Plot Hard ML decoder (if available)
if ber_ml_hard is not None:
    plt.plot(ebno, ber_ml_hard, 
             marker="o", 
             markersize=5,
             color=colors["hard_ml"], 
             linewidth=2.2,
             label="ML (Hard Decision)",
             markerfacecolor=colors["hard_ml"],
             markeredgecolor=colors["hard_ml"])

# Plot Soft ML decoder
plt.plot(ebno, ber_ml_soft, 
         marker="o", 
         markersize=5,
         color=colors["soft_ml"], 
         linewidth=2.2,
         label="ML (Soft Decision)",
         markerfacecolor=colors["soft_ml"],
         markeredgecolor=colors["soft_ml"])

# Set log scale for y-axis
plt.yscale("log")

# Labels and title
plt.xlabel("Eb/N0 (dB)", fontsize=12)
plt.ylabel("Bit Error Rate", fontsize=12)
plt.title("Hard vs Soft ML Decoder Performance", fontsize=14, fontweight='bold')

# Grid styling: very light gray
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6, color="#E0E0E0")

# Legend
plt.legend(fontsize=11, framealpha=0.9)

# Set axis limits
plt.xlim(left=ebno.min() - 0.5, right=ebno.max() + 0.5)

# Tight layout
plt.tight_layout()

# Save figure
output_path = project_root / "data" / "hard_vs_soft_ml.png"
plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
print(f"✓ Slide 9 plot saved to {output_path}")

plt.close()

