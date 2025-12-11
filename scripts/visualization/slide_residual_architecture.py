#!/usr/bin/env python3
"""
Residual Architecture Block Diagram

Generates a simple FC stack with a residual skip for presentation slides.
"""

import matplotlib.pyplot as plt
from pathlib import Path

# Palette
colors = {
    "text": "#4A4A4A",           # dark gray for text/arrows
    "box": "#E8F5E9",            # light green for layer boxes
    "residual": "#00C853",       # bright green for skip
}

project_root = Path(__file__).parent.parent.parent
output_path = project_root / "data" / "residual_architecture.png"

plt.figure(figsize=(8, 3), facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")

# Text blocks
plt.text(0.1, 0.5, "LLR Input (7)", fontsize=12, ha="center", color=colors["text"])
plt.text(0.3, 0.5, "FC 128\nReLU", fontsize=12, ha="center",
         bbox=dict(boxstyle="round,pad=0.4", fc=colors["box"], ec=colors["text"], linewidth=2),
         color=colors["text"])
plt.text(0.5, 0.5, "FC 64\nReLU", fontsize=12, ha="center",
         bbox=dict(boxstyle="round,pad=0.4", fc=colors["box"], ec=colors["text"], linewidth=2),
         color=colors["text"])
plt.text(0.7, 0.5, "FC 32\nReLU", fontsize=12, ha="center",
         bbox=dict(boxstyle="round,pad=0.4", fc=colors["box"], ec=colors["text"], linewidth=2),
         color=colors["text"])
plt.text(0.9, 0.5, "Output (4 bits)", fontsize=12, ha="center", color=colors["text"])

# Arrow style
arrow_kwargs = dict(arrowstyle="-|>", color=colors["text"], linewidth=2.2, mutation_scale=14)

# Forward arrows
plt.annotate("", xy=(0.23, 0.5), xytext=(0.15, 0.5), arrowprops=arrow_kwargs)
plt.annotate("", xy=(0.43, 0.5), xytext=(0.35, 0.5), arrowprops=arrow_kwargs)
plt.annotate("", xy=(0.63, 0.5), xytext=(0.55, 0.5), arrowprops=arrow_kwargs)

# Residual skip
skip_kwargs = dict(arrowstyle="-|>", color=colors["residual"], linewidth=2.2, mutation_scale=12)
plt.annotate("", xy=(0.63, 0.58), xytext=(0.14, 0.58), arrowprops=skip_kwargs)

plt.axis("off")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"✓ Residual architecture diagram saved to {output_path}")
plt.close()
