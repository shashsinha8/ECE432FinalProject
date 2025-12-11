# Presentation Visualization Scripts

This directory contains scripts to generate presentation-ready visualizations for the Hamming(7,4) decoder project.

## Color Palette

All visualizations use a consistent color palette:

- **Classical**: `#4A4A4A` (dark gray)
- **Hard ML**: `#9E9E9E` (light gray)
- **Soft ML**: `#2E7D32` (dark green)
- **Deep**: `#1B5E20` (deep green)
- **Wide**: `#66BB6A` (mid green)
- **Residual**: `#00C853` (bright green)

## Styling

All plots follow consistent styling:
- **Background**: White
- **Grid**: Very light gray (`#E0E0E0`)
- **Line width**: 2.2px
- **Markers**: Small filled circles (size 5)
- **Resolution**: 300 DPI

## Scripts

### `slide6_classical_ber.py`

Generates **Slide 6: Classical Decoder BER Curve**

**Output**: `data/classical_ber.png`

**Usage**:
```bash
python scripts/visualization/slide6_classical_ber.py
```

### `generate_all_slides.py`

Generates all presentation visualizations:

1. **Slide 6**: Classical Decoder BER Curve
2. **Comparison**: Classical vs Soft-Decision ML
3. **Comprehensive**: All Decoders Comparison
4. **Phase 4**: Architecture Variants Comparison

**Usage**:
```bash
python scripts/visualization/generate_all_slides.py
```

**Outputs**:
- `data/classical_ber.png` - Classical decoder performance
- `data/comparison_classical_vs_soft.png` - Classical vs Soft ML comparison
- `data/comparison_all_decoders_presentation.png` - All decoders comparison
- `data/phase4_architectures_comparison.png` - Phase 4 architecture variants

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Project data file: `data/phase5_final_results.npy`

## Notes

- All plots use log scale for BER (y-axis)
- Plots are optimized for presentation slides (7x5 or 10x6 inches)
- High-resolution output (300 DPI) for crisp printing
- White background for clean presentation appearance

