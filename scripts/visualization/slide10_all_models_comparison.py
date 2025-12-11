#!/usr/bin/env python3
"""
Slide 10: Full Comparison - Classical vs Standard vs Deep vs Wide vs Residual

Generates a presentation-ready BER comparison across the classical decoder
and all ML architectures (standard/soft, deep, wide, residual).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Color palette
colors = {
    "classical": "#4A4A4A",      # dark gray
    "standard": "#2E7D32",       # dark green (soft ML / standard)
    "deep": "#1B5E20",           # deep green
    "wide": "#66BB6A",           # mid green
    "residual": "#00C853",       # bright green
}

# Styling
GRID_COLOR = "#E0E0E0"
LINE_WIDTH = 2.2
MARKER_SIZE = 5
FIG_SIZE = (7, 5)

# Load results
results_path = project_root / "data" / "phase5_final_results.npy"
results = np.load(results_path, allow_pickle=True).item()

# Extract ebno and BER arrays from nested structure
classical = results["results"]["classical"]
ebno = classical["ebno_range"]
ber_classical = classical["ber"]

standard = results["results"].get("soft_ml") or results["results"].get("phase3")
if standard is None:
    raise ValueError("Standard (soft ML) results not found in results file.")
ber_standard = standard["ber"]

deep = results["results"].get("phase4_deep")
wide = results["results"].get("phase4_wide")
residual = results["results"].get("phase4_residual")

plt.figure(figsize=FIG_SIZE, facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Plot helpers
plt.plot(ebno, ber_classical, marker="o", markersize=MARKER_SIZE,
         color=colors["classical"], linewidth=LINE_WIDTH,
         label="Classical",
         markerfacecolor=colors["classical"], markeredgecolor=colors["classical"])

plt.plot(ebno, ber_standard, marker="o", markersize=MARKER_SIZE,
         color=colors["standard"], linewidth=LINE_WIDTH,
         label="Standard ML",
         markerfacecolor=colors["standard"], markeredgecolor=colors["standard"])

if deep is not None:
    plt.plot(ebno, deep["ber"], marker="o", markersize=MARKER_SIZE,
             color=colors["deep"], linewidth=LINE_WIDTH,
             label="Deep ML",
             markerfacecolor=colors["deep"], markeredgecolor=colors["deep"])
if wide is not None:
    plt.plot(ebno, wide["ber"], marker="o", markersize=MARKER_SIZE,
             color=colors["wide"], linewidth=LINE_WIDTH,
             label="Wide ML",
             markerfacecolor=colors["wide"], markeredgecolor=colors["wide"])
if residual is not None:
    plt.plot(ebno, residual["ber"], marker="o", markersize=MARKER_SIZE,
             color=colors["residual"], linewidth=LINE_WIDTH,
             label="Residual ML (Best)",
             markerfacecolor=colors["residual"], markeredgecolor=colors["residual"])

plt.yscale("log")
plt.xlabel("Eb/N0 (dB)", fontsize=12)
plt.ylabel("Bit Error Rate", fontsize=12)
plt.title("Performance Comparison: Classical vs ML Decoders", fontsize=14, fontweight='bold')
plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6, color=GRID_COLOR)
plt.legend(fontsize=11, framealpha=0.9)
plt.xlim(left=ebno.min() - 0.5, right=ebno.max() + 0.5)

plt.tight_layout()
output_path = project_root / "data" / "all_models_comparison.png"
plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
print(f"✓ Slide 10 plot saved to {output_path}")
plt.close()
