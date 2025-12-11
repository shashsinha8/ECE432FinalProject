#!/usr/bin/env python3
"""
Run baseline performance evaluation for classical Hamming decoder.

This script evaluates the classical Hamming(7,4) decoder performance
across a range of Eb/N0 values and generates BER vs. Eb/N0 plots.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.evaluation import run_baseline_evaluation

if __name__ == "__main__":
    # Run baseline evaluation
    # Parameters can be adjusted as needed
    ebno_range, ber_values, error_counts, total_bits = run_baseline_evaluation(
        ebno_start=-5,      # Starting Eb/N0 in dB
        ebno_end=10,        # Ending Eb/N0 in dB
        num_points=16,      # Number of Eb/N0 points
        num_bits_per_point=100000,  # Bits per point (100k for good statistics)
        seed=42,            # Random seed for reproducibility
        save_plot=True      # Save plot to data/baseline_ber_curve.png
    )
    
    print("=" * 60)
    print("Baseline evaluation complete!")
    print("=" * 60)
    print(f"Plot saved to: data/baseline_ber_curve.png")
    print()
    print("Next steps:")
    print("  - Review the BER curve to verify expected Hamming code performance")
    print("  - Proceed to Phase 4: ML Model Development")

