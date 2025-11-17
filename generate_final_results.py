#!/usr/bin/env python3
"""
Generate Final Results: Complete Comparison of Classical vs. ML Decoders

This script generates the final comparison plots and results for the project.
It evaluates both classical and ML-assisted decoders and creates comprehensive
visualizations and analysis.
"""

import argparse
import numpy as np
from src.evaluation import (
    evaluate_classical_decoder,
    generate_ebno_range,
    run_baseline_evaluation
)
from src.ml_evaluation import (
    run_ml_evaluation,
    compare_decoders
)
import os


def main():
    parser = argparse.ArgumentParser(description='Generate final comparison results')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained ML model (optional)')
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
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device for inference')
    parser.add_argument('--skip_ml', action='store_true',
                       help='Skip ML evaluation (classical only)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Final Results Generation: Classical vs. ML-Assisted Decoder")
    print("=" * 70)
    print()
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Generate Eb/N0 range
    ebno_range = generate_ebno_range(args.ebno_start, args.ebno_end,
                                    num_points=args.num_points)
    
    # Evaluate classical decoder
    print("Step 1: Evaluating Classical Decoder")
    print("-" * 70)
    classical_ber, classical_errors, classical_total = evaluate_classical_decoder(
        ebno_range, args.num_bits, seed=args.seed
    )
    
    # Save classical results
    classical_results_path = 'data/classical_decoder_results.npy'
    np.save(classical_results_path, {
        'ebno_range': ebno_range,
        'ber': classical_ber,
        'errors': classical_errors,
        'total_bits': classical_total
    })
    print(f"Classical decoder results saved to: {classical_results_path}")
    print()
    
    # Evaluate ML decoder if model provided
    ml_ber = None
    if args.model_path and not args.skip_ml:
        if os.path.exists(args.model_path):
            print("Step 2: Evaluating ML-Assisted Decoder")
            print("-" * 70)
            ml_ebno, ml_ber, ml_errors, ml_total = run_ml_evaluation(
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
            
            # Save ML results
            ml_results_path = f'data/ml_decoder_{args.approach}_results.npy'
            np.save(ml_results_path, {
                'ebno_range': ml_ebno,
                'ber': ml_ber,
                'errors': ml_errors,
                'total_bits': ml_total
            })
            print(f"ML decoder results saved to: {ml_results_path}")
            print()
        else:
            print(f"Warning: Model file not found: {args.model_path}")
            print("Skipping ML evaluation.")
            print()
    
    # Generate comparison plot if both available
    if ml_ber is not None:
        print("Step 3: Generating Comparison Plot")
        print("-" * 70)
        compare_decoders(
            ebno_range, classical_ber, ml_ber,
            ml_label=f"ML Decoder ({args.approach})",
            save_path=f'data/final_comparison_{args.approach}.png',
            show_plot=False
        )
        print(f"Comparison plot saved to: data/final_comparison_{args.approach}.png")
        print()
        
        # Calculate and print improvement statistics
        print("Step 4: Performance Analysis")
        print("-" * 70)
        print("Eb/N0 (dB) | Classical BER | ML BER      | Improvement | Coding Gain")
        print("-" * 70)
        
        for i in range(len(ebno_range)):
            c_ber = classical_ber[i]
            m_ber = ml_ber[i]
            
            if c_ber > 0 and m_ber > 0:
                improvement = (c_ber - m_ber) / c_ber * 100
                # Coding gain: difference in Eb/N0 needed for same BER
                # (simplified calculation)
                if i > 0 and classical_ber[i-1] > 0:
                    gain_estimate = "~" + str(round((ebno_range[i] - ebno_range[i-1]) * improvement / 100, 2)) + " dB"
                else:
                    gain_estimate = "N/A"
            else:
                improvement = 0.0
                gain_estimate = "N/A"
            
            print(f"  {ebno_range[i]:5.1f}   | {c_ber:11.2e} | {m_ber:10.2e} | "
                  f"{improvement:10.1f}% | {gain_estimate}")
        
        print()
        
        # Summary statistics
        valid_indices = (classical_ber > 0) & (ml_ber > 0)
        if np.any(valid_indices):
            avg_improvement = np.mean((classical_ber[valid_indices] - ml_ber[valid_indices]) / 
                                     classical_ber[valid_indices] * 100)
            max_improvement = np.max((classical_ber[valid_indices] - ml_ber[valid_indices]) / 
                                   classical_ber[valid_indices] * 100)
            
            print("Summary Statistics:")
            print(f"  Average BER improvement: {avg_improvement:.1f}%")
            print(f"  Maximum BER improvement: {max_improvement:.1f}%")
            print()
    
    # Generate summary report
    print("Step 5: Generating Summary Report")
    print("-" * 70)
    report_path = 'data/final_results_summary.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ML-Assisted Hamming Code Decoder - Final Results Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Evaluation Parameters:\n")
        f.write(f"  Eb/N0 range: {args.ebno_start} to {args.ebno_end} dB\n")
        f.write(f"  Number of points: {args.num_points}\n")
        f.write(f"  Bits per point: {args.num_bits:,}\n")
        f.write(f"  Random seed: {args.seed}\n")
        f.write(f"  Total bits simulated: {np.sum(classical_total):,}\n\n")
        
        f.write("Classical Decoder Results:\n")
        f.write("-" * 70 + "\n")
        f.write("Eb/N0 (dB) | BER      | Errors\n")
        f.write("-" * 70 + "\n")
        for i in range(len(ebno_range)):
            f.write(f"  {ebno_range[i]:5.1f}   | {classical_ber[i]:.2e} | {classical_errors[i]:,}\n")
        f.write("\n")
        
        if ml_ber is not None:
            f.write(f"ML Decoder Results ({args.approach} approach):\n")
            f.write("-" * 70 + "\n")
            f.write("Eb/N0 (dB) | BER      | Errors\n")
            f.write("-" * 70 + "\n")
            for i in range(len(ebno_range)):
                f.write(f"  {ebno_range[i]:5.1f}   | {ml_ber[i]:.2e} | {ml_errors[i]:,}\n")
            f.write("\n")
    
    print(f"Summary report saved to: {report_path}")
    print()
    
    print("=" * 70)
    print("Final Results Generation: COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  - Classical decoder results: data/classical_decoder_results.npy")
    if ml_ber is not None:
        print(f"  - ML decoder results: data/ml_decoder_{args.approach}_results.npy")
        print(f"  - Comparison plot: data/final_comparison_{args.approach}.png")
    print(f"  - Summary report: data/final_results_summary.txt")
    print()
    print("All results are reproducible with seed =", args.seed)


if __name__ == "__main__":
    main()

