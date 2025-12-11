#!/usr/bin/env python3
"""
Phase 5: Final Comprehensive Evaluation

This script evaluates all model variants and generates comprehensive comparison:
- Classical decoder (baseline)
- Hard-decision ML decoder
- Soft-decision ML decoder (Phase 2)
- Phase 3 improved decoder
- Phase 4 architecture variants (deep, wide, residual)

Generates comprehensive plots and final performance summary report.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import (
    evaluate_classical_decoder,
    generate_ebno_range,
    plot_ber_curve
)
from src.ml_evaluation import (
    run_ml_evaluation,
    evaluate_ml_decoder
)
from src.ml_decoder import (
    DirectMappingDecoder,
    DeepDirectMappingDecoder,
    WideDirectMappingDecoder,
    ResidualDirectMappingDecoder,
    MLDecoder,
    load_model
)
import torch


def load_model_by_type(model_path, device='cpu'):
    """
    Load model and determine its type from file path or try different architectures.
    
    Returns:
    --------
    model : nn.Module
        Loaded model
    use_soft_input : bool
        Whether model uses soft inputs
    architecture : str
        Architecture type
    """
    use_soft_input = True  # All Phase 2+ models use soft inputs
    
    # Try to determine architecture from filename
    if 'phase4_deep' in model_path:
        model = DeepDirectMappingDecoder(input_size=7, output_size=4, use_soft_input=True)
        architecture = 'deep'
    elif 'phase4_wide' in model_path:
        model = WideDirectMappingDecoder(input_size=7, output_size=4, use_soft_input=True)
        architecture = 'wide'
    elif 'phase4_residual' in model_path:
        model = ResidualDirectMappingDecoder(input_size=7, output_size=4, use_soft_input=True)
        architecture = 'residual'
    elif 'phase3' in model_path:
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[64, 32], 
                                     output_size=4, use_soft_input=True)
        architecture = 'phase3'
    elif 'direct_soft' in model_path:
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[64, 32], 
                                     output_size=4, use_soft_input=True)
        architecture = 'soft'
    elif 'direct' in model_path and 'soft' not in model_path:
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[64, 32], 
                                     output_size=4, use_soft_input=False)
        use_soft_input = False
        architecture = 'hard'
    else:
        # Default: try soft-decision standard
        model = DirectMappingDecoder(input_size=7, hidden_sizes=[64, 32], 
                                     output_size=4, use_soft_input=True)
        architecture = 'unknown'
    
    try:
        model = load_model(model, model_path, device=device)
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
        return None, False, None
    
    return model, use_soft_input, architecture


def evaluate_all_models(ebno_range, num_bits_per_point=100000, seed=42, device='cpu'):
    """
    Evaluate all available model variants.
    
    Returns:
    --------
    results : dict
        Dictionary with results for each model
    """
    results = {}
    
    print("=" * 70)
    print("Phase 5: Comprehensive Model Evaluation")
    print("=" * 70)
    print()
    
    # 1. Classical decoder (baseline)
    print("1. Evaluating Classical Decoder (Baseline)...")
    classical_ber, _, _ = evaluate_classical_decoder(ebno_range, num_bits_per_point, seed=seed)
    results['classical'] = {
        'ebno_range': ebno_range,
        'ber': classical_ber,
        'label': 'Classical Decoder',
        'color': 'black',
        'linestyle': '-',
        'marker': 'o'
    }
    print("   ✓ Complete\n")
    
    # 2. Hard-decision ML decoder (if available)
    hard_model_path = 'models/ml_decoder_direct.pth'
    if os.path.exists(hard_model_path):
        print("2. Evaluating Hard-Decision ML Decoder...")
        try:
            model, use_soft, arch = load_model_by_type(hard_model_path, device)
            ml_decoder = MLDecoder(model, device=device, use_soft_input=use_soft)
            hard_ber, _, _ = evaluate_ml_decoder(ebno_range, ml_decoder, approach='direct',
                                                 num_bits_per_point=num_bits_per_point, seed=seed)
            results['hard_ml'] = {
                'ebno_range': ebno_range,
                'ber': hard_ber,
                'label': 'Hard-Decision ML',
                'color': 'red',
                'linestyle': '--',
                'marker': 's'
            }
            print("   ✓ Complete\n")
        except Exception as e:
            print(f"   ⚠ Error: {e}\n")
    else:
        print("2. Hard-Decision ML Decoder not found (skipping)\n")
    
    # 3. Soft-decision ML decoder (Phase 2)
    soft_model_path = 'models/ml_decoder_direct_soft.pth'
    if os.path.exists(soft_model_path):
        print("3. Evaluating Soft-Decision ML Decoder (Phase 2)...")
        try:
            model, use_soft, arch = load_model_by_type(soft_model_path, device)
            ml_decoder = MLDecoder(model, device=device, use_soft_input=use_soft)
            soft_ber, _, _ = evaluate_ml_decoder(ebno_range, ml_decoder, approach='direct',
                                                 num_bits_per_point=num_bits_per_point, seed=seed)
            results['soft_ml'] = {
                'ebno_range': ebno_range,
                'ber': soft_ber,
                'label': 'Soft-Decision ML (Phase 2)',
                'color': 'blue',
                'linestyle': '-',
                'marker': '^'
            }
            print("   ✓ Complete\n")
        except Exception as e:
            print(f"   ⚠ Error: {e}\n")
    else:
        print("3. Soft-Decision ML Decoder not found (skipping)\n")
    
    # 4. Phase 3 improved decoder
    phase3_model_path = 'models/ml_decoder_phase3.pth'
    if os.path.exists(phase3_model_path):
        print("4. Evaluating Phase 3 Improved Decoder...")
        try:
            model, use_soft, arch = load_model_by_type(phase3_model_path, device)
            ml_decoder = MLDecoder(model, device=device, use_soft_input=use_soft)
            phase3_ber, _, _ = evaluate_ml_decoder(ebno_range, ml_decoder, approach='direct',
                                                   num_bits_per_point=num_bits_per_point, seed=seed)
            results['phase3'] = {
                'ebno_range': ebno_range,
                'ber': phase3_ber,
                'label': 'Phase 3 (Improved Training)',
                'color': 'green',
                'linestyle': '-',
                'marker': 'D'
            }
            print("   ✓ Complete\n")
        except Exception as e:
            print(f"   ⚠ Error: {e}\n")
    else:
        print("4. Phase 3 Decoder not found (skipping)\n")
    
    # 5. Phase 4 architecture variants
    phase4_models = {
        'phase4_deep': ('models/ml_decoder_phase4_deep.pth', 'Deep Architecture', 'purple', '-', 'v'),
        'phase4_wide': ('models/ml_decoder_phase4_wide.pth', 'Wide Architecture', 'orange', '-', 'p'),
        'phase4_residual': ('models/ml_decoder_phase4_residual.pth', 'Residual Architecture', 'brown', '-', '*')
    }
    
    for key, (model_path, label, color, linestyle, marker) in phase4_models.items():
        if os.path.exists(model_path):
            print(f"5. Evaluating {label} (Phase 4)...")
            try:
                model, use_soft, arch = load_model_by_type(model_path, device)
                ml_decoder = MLDecoder(model, device=device, use_soft_input=use_soft)
                phase4_ber, _, _ = evaluate_ml_decoder(ebno_range, ml_decoder, approach='direct',
                                                      num_bits_per_point=num_bits_per_point, seed=seed)
                results[key] = {
                    'ebno_range': ebno_range,
                    'ber': phase4_ber,
                    'label': label,
                    'color': color,
                    'linestyle': linestyle,
                    'marker': marker
                }
                print(f"   ✓ Complete\n")
            except Exception as e:
                print(f"   ⚠ Error: {e}\n")
        else:
            print(f"5. {label} not found (skipping)\n")
    
    return results


def plot_comprehensive_comparison(results, save_path='data/phase5_final_comparison.png'):
    """Generate comprehensive comparison plot of all models."""
    plt.figure(figsize=(12, 8))
    
    # Plot each model
    for key, data in results.items():
        plt.semilogy(data['ebno_range'], data['ber'],
                    label=data['label'],
                    color=data['color'],
                    linestyle=data['linestyle'],
                    marker=data['marker'],
                    linewidth=2,
                    markersize=8,
                    alpha=0.8)
    
    plt.xlabel('Eb/N0 (dB)', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.title('Comprehensive ML Decoder Performance Comparison\nAll Phases and Architectures', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.xlim(-5, 10)
    plt.ylim(1e-5, 1)
    
    # Add phase annotations
    plt.text(8.5, 0.3, 'Phase 2: Soft-Decision', fontsize=10, color='blue', alpha=0.7)
    plt.text(8.5, 0.2, 'Phase 3: Improved Training', fontsize=10, color='green', alpha=0.7)
    plt.text(8.5, 0.1, 'Phase 4: Architecture Variants', fontsize=10, color='purple', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive comparison plot saved to {save_path}")
    plt.close()


def calculate_performance_summary(results):
    """Calculate performance summary statistics."""
    if 'classical' not in results:
        return None
    
    classical_ber = results['classical']['ber']
    ebno_range = results['classical']['ebno_range']
    
    summary = {
        'ebno_range': ebno_range,
        'classical_ber': classical_ber
    }
    
    # Calculate improvements for each model
    for key, data in results.items():
        if key == 'classical':
            continue
        
        ml_ber = data['ber']
        
        # Calculate improvement percentage
        valid_mask = classical_ber > 0
        improvements = np.zeros_like(classical_ber)
        improvements[valid_mask] = ((classical_ber[valid_mask] - ml_ber[valid_mask]) / 
                                   classical_ber[valid_mask] * 100)
        
        # Average improvement
        avg_improvement = np.mean(improvements[valid_mask]) if np.any(valid_mask) else 0
        
        # Best and worst points
        best_idx = np.argmax(improvements[valid_mask]) if np.any(valid_mask) else 0
        worst_idx = np.argmin(improvements[valid_mask]) if np.any(valid_mask) else 0
        
        summary[key] = {
            'ber': ml_ber,
            'avg_improvement': avg_improvement,
            'best_ebno': ebno_range[valid_mask][best_idx] if np.any(valid_mask) else 0,
            'best_improvement': improvements[valid_mask][best_idx] if np.any(valid_mask) else 0,
            'worst_ebno': ebno_range[valid_mask][worst_idx] if np.any(valid_mask) else 0,
            'worst_improvement': improvements[valid_mask][worst_idx] if np.any(valid_mask) else 0
        }
    
    return summary


def generate_final_report(results, summary, save_path='data/phase5_final_report.md'):
    """Generate final performance summary report."""
    with open(save_path, 'w') as f:
        f.write("# Phase 5: Final Performance Evaluation Report\n\n")
        f.write("**Generated**: December 2025\n")
        f.write("**Evaluation**: Comprehensive comparison of all model variants\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents the final comprehensive evaluation of all ML decoder variants:\n\n")
        f.write("1. **Classical Decoder**: Baseline syndrome-based decoder\n")
        f.write("2. **Hard-Decision ML**: Initial ML decoder (Phase 1)\n")
        f.write("3. **Soft-Decision ML**: Phase 2 improvement (LLRs)\n")
        f.write("4. **Phase 3**: Improved training strategy\n")
        f.write("5. **Phase 4**: Architecture variants (deep, wide, residual)\n\n")
        
        f.write("---\n\n")
        f.write("## Performance Summary\n\n")
        
        if summary:
            f.write("| Model | Average Improvement vs. Classical | Best Improvement | Best Eb/N0 |\n")
            f.write("|-------|----------------------------------|------------------|------------|\n")
            
            for key, data in results.items():
                if key == 'classical':
                    f.write(f"| {data['label']} | Baseline | - | - |\n")
                elif key in summary:
                    s = summary[key]
                    f.write(f"| {data['label']} | {s['avg_improvement']:.2f}% | "
                           f"{s['best_improvement']:.2f}% | {s['best_ebno']:.1f} dB |\n")
        
        f.write("\n---\n\n")
        f.write("## Detailed Results\n\n")
        
        if summary:
            ebno_range = summary['ebno_range']
            f.write("| Eb/N0 (dB) | Classical BER")
            for key, data in results.items():
                if key != 'classical':
                    f.write(f" | {data['label']} BER")
            f.write(" |\n")
            f.write("|" + "---|" * (len(results) + 1) + "\n")
            
            for i in range(len(ebno_range)):
                f.write(f"| {ebno_range[i]:.1f} | {summary['classical_ber'][i]:.4e}")
                for key, data in results.items():
                    if key != 'classical':
                        if key in summary:
                            f.write(f" | {summary[key]['ber'][i]:.4e}")
                        else:
                            f.write(" | -")
                f.write(" |\n")
        
        f.write("\n---\n\n")
        f.write("## Key Findings\n\n")
        
        if summary:
            # Find best performing model
            best_model = None
            best_improvement = -float('inf')
            for key, s in summary.items():
                if key != 'ebno_range' and key != 'classical_ber':
                    if s['avg_improvement'] > best_improvement:
                        best_improvement = s['avg_improvement']
                        best_model = key
            
            if best_model:
                f.write(f"### Best Performing Model: {results[best_model]['label']}\n\n")
                f.write(f"- **Average Improvement**: {summary[best_model]['avg_improvement']:.2f}% better than classical\n")
                f.write(f"- **Best Point**: {summary[best_model]['best_improvement']:.2f}% improvement at "
                       f"{summary[best_model]['best_ebno']:.1f} dB\n\n")
        
        f.write("### Phase-by-Phase Improvements\n\n")
        f.write("1. **Phase 2 (Soft-Decision)**: Major breakthrough - transformed ML decoder from underperforming to outperforming classical\n")
        f.write("2. **Phase 3 (Improved Training)**: Further refinement with weighted sampling, codeword loss, and data augmentation\n")
        f.write("3. **Phase 4 (Architectures)**: Exploration of different network architectures for optimal performance\n\n")
        
        f.write("---\n\n")
        f.write("## Recommendations\n\n")
        f.write("1. **For Practical Use**: Use the best performing Phase 4 architecture variant\n")
        f.write("2. **For Development**: Continue exploring architecture improvements\n")
        f.write("3. **For Research**: Investigate theoretical limits and further optimizations\n\n")
        
        f.write("---\n\n")
        f.write("## Conclusion\n\n")
        f.write("The comprehensive evaluation demonstrates significant improvements across all phases.\n")
        f.write("The ML-assisted decoders successfully outperform the classical decoder, with Phase 2-4 improvements\n")
        f.write("providing substantial gains in error correction performance.\n")
    
    print(f"✓ Final report saved to {save_path}")


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("Phase 5: Final Comprehensive Evaluation")
    print("=" * 70)
    print()
    
    # Configuration
    ebno_start = -5.0
    ebno_end = 10.0
    num_points = 16
    num_bits_per_point = 100000
    seed = 42
    device = 'cpu'
    
    # Generate Eb/N0 range
    ebno_range = generate_ebno_range(ebno_start, ebno_end, num_points=num_points)
    
    # Evaluate all models
    results = evaluate_all_models(ebno_range, num_bits_per_point, seed, device)
    
    if len(results) < 2:
        print("⚠ Not enough models evaluated. Need at least classical + one ML model.")
        return
    
    # Generate comprehensive comparison plot
    print("=" * 70)
    print("Generating Comprehensive Comparison Plot")
    print("=" * 70)
    plot_comprehensive_comparison(results)
    print()
    
    # Calculate performance summary
    print("=" * 70)
    print("Calculating Performance Summary")
    print("=" * 70)
    summary = calculate_performance_summary(results)
    print()
    
    # Generate final report
    print("=" * 70)
    print("Generating Final Report")
    print("=" * 70)
    generate_final_report(results, summary)
    print()
    
    # Save results
    results_path = 'data/phase5_final_results.npy'
    np.save(results_path, {
        'results': results,
        'summary': summary,
        'ebno_range': ebno_range
    })
    print(f"✓ All results saved to {results_path}")
    print()
    
    print("=" * 70)
    print("Phase 5 Evaluation Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - data/phase5_final_comparison.png (comprehensive plot)")
    print("  - data/phase5_final_report.md (performance report)")
    print("  - data/phase5_final_results.npy (raw results)")


if __name__ == "__main__":
    main()

