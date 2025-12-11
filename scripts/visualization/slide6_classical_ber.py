#!/usr/bin/env python3
"""
Slide 6: Classical Decoder BER Curve

Generates a visualization of the Classical Hamming(7,4) decoder performance
with presentation-ready styling.
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

# Load classical decoder results
results_path = project_root / "data" / "phase5_final_results.npy"
results = np.load(results_path, allow_pickle=True).item()

# Extract classical decoder data
classical_data = results["results"]["classical"]
ebno = classical_data["ebno_range"]
classical_ber = classical_data["ber"]

# Create figure with white background
plt.figure(figsize=(7, 5), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Plot classical decoder BER curve
plt.plot(ebno, classical_ber, 
         marker="o", 
         markersize=5,
         color=colors["classical"], 
         linewidth=2.2,
         markerfacecolor=colors["classical"],
         markeredgecolor=colors["classical"])

# Set log scale for y-axis
plt.yscale("log")

# Labels and title
plt.xlabel("Eb/N0 (dB)", fontsize=12)
plt.ylabel("Bit Error Rate (log scale)", fontsize=12)
plt.title("Classical Hamming(7,4) Decoder Performance", fontsize=14, fontweight='bold')

# Grid styling: very light gray
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6, color="#E0E0E0")

# Set axis limits if needed
plt.xlim(left=ebno.min() - 0.5, right=ebno.max() + 0.5)
plt.ylim(bottom=max(1e-5, classical_ber.min() * 0.5), top=min(1.0, classical_ber.max() * 2))

# Tight layout
plt.tight_layout()

# Save figure
output_path = project_root / "data" / "classical_ber.png"
plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
print(f"✓ Slide 6 plot saved to {output_path}")

plt.close()

