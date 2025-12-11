#!/usr/bin/env python3
"""
Compare Classical and ML-Assisted Decoder Performance

This script evaluates both classical and ML-assisted decoders and creates
comparison plots.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.evaluation import (
    evaluate_classical_decoder,
    generate_ebno_range
)
from src.ml_evaluation import (
    run_ml_evaluation,
    compare_decoders
)


def main():
    parser = argparse.ArgumentParser(description='Compare classical and ML decoders')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained ML model')
    parser.add_argument('--approach', type=str, choices=['direct', 'post'],
                       default='direct', help='ML decoder approach')
    parser.add_argument('--ebno_start', type=float, default=-5.0,
                       help='Starting Eb/N0 (dB)')
    parser.add_argument('--ebno_end', type=float, default=10.0,
                       help='Ending Eb/N0 (dB)')
    parser.add_argument('--num_points', type=int, default=16,
                       help='Number of Eb/N0 points')
    parser.add_argument('--num_bits', type=int, default=100000,
                       help='Number of bits per point')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device for inference')
    parser.add_argument('--skip_classical', action='store_true',
                       help='Skip classical decoder evaluation (use existing results)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Decoder Performance Comparison")
    print("=" * 60)
    print()
    
    # Generate Eb/N0 range
    ebno_range = generate_ebno_range(args.ebno_start, args.ebno_end, 
                                    num_points=args.num_points)
    
    # Evaluate classical decoder
    if not args.skip_classical:
        print("Evaluating classical decoder...")
        print()
        classical_ber, _, _ = evaluate_classical_decoder(
            ebno_range, args.num_bits, seed=args.seed
        )
        print()
    else:
        print("Skipping classical decoder evaluation (using existing results)")
        print("Note: You should have classical BER results available")
        classical_ber = None
    
    # Evaluate ML decoder
    print("Evaluating ML-assisted decoder...")
    print()
    ml_ebno, ml_ber, _, _ = run_ml_evaluation(
        model_path=args.model_path,
        approach=args.approach,
        ebno_start=args.ebno_start,
        ebno_end=args.ebno_end,
        num_points=args.num_points,
        num_bits_per_point=args.num_bits,
        seed=args.seed,
        device=args.device,
        save_plot=True
    )
    
    # Compare if classical results available
    if classical_ber is not None:
        print()
        print("=" * 60)
        print("Creating Comparison Plot")
        print("=" * 60)
        compare_decoders(
            ebno_range, classical_ber, ml_ber,
            ml_label=f"ML Decoder ({args.approach})",
            save_path=f"data/decoder_comparison_{args.approach}.png",
            show_plot=False
        )
        print(f"Comparison plot saved to: data/decoder_comparison_{args.approach}.png")
        print()
        
        # Calculate improvement
        print("Performance Comparison:")
        print("Eb/N0 (dB) | Classical BER | ML BER      | Improvement")
        print("-" * 60)
        for i in range(len(ebno_range)):
            classical_val = classical_ber[i]
            ml_val = ml_ber[i]
            if classical_val > 0:
                improvement = (classical_val - ml_val) / classical_val * 100
                print(f"  {ebno_range[i]:5.1f}   | {classical_val:.2e}   | {ml_val:.2e} | {improvement:+.1f}%")
            else:
                print(f"  {ebno_range[i]:5.1f}   | {classical_val:.2e}   | {ml_val:.2e} | N/A")
    
    print()
    print("=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

