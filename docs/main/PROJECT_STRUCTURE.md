# Project Structure

This document describes the organized structure of the ECE 432 Final Project.

## Essential Files

### Core Source Code (`src/`)
- `hamming.py` - Hamming(7,4) encoder/decoder
- `classical_decoder.py` - Classical syndrome-based decoder
- `channel.py` - BPSK modulation and AWGN channel
- `evaluation.py` - BER evaluation framework
- `ml_decoder.py` - ML model architectures
- `ml_evaluation.py` - ML decoder evaluation

### Tests (`tests/`)
- All unit tests (66 tests total)
- Comprehensive test coverage

### Training Scripts
- `train_ml_decoder_soft.py` - Soft-decision ML training (recommended)
- `train_ml_decoder_phase3.py` - Improved training with weighted sampling
- `train_ml_decoder_phase4.py` - Architecture variants (deep, wide, residual)

### Evaluation Scripts
- `run_baseline_evaluation.py` - Classical decoder baseline
- `compare_decoders.py` - Two-model comparison
- `final_evaluation_phase5.py` - Comprehensive evaluation (all models)

### Documentation
- `README.md` - Main project documentation
- `PROJECT_EXPLANATION.txt` - Complete project explanation
- `PROJECT_SUMMARY.md` - Project summary
- `VERIFICATION_REPORT.txt` - Verification of results legitimacy
- `docs/CODEBASE_DOCUMENTATION.md` - Detailed technical documentation
- `docs/SOFT_DECISION_ANALYSIS.md` - Soft-decision analysis
- `docs/PHASE3_IMPROVEMENTS.md` - Training improvements
- `docs/PHASE4_ARCHITECTURES.md` - Architecture details
- `docs/PHASE5_FINAL_EVALUATION.md` - Final evaluation

### Results (`data/`)
- `phase5_final_comparison.png` - Comprehensive comparison plot
- `phase5_final_report.md` - Performance summary
- `phase5_final_results.npy` - Raw results data
- `baseline_ber_curve.png` - Classical decoder baseline
- `comparison_all_decoders.png` - All decoders comparison
- `training_comparison_soft_vs_hard.png` - Training comparison

### Models (`models/`)
- `ml_decoder_direct_soft.pth` - Soft-decision ML model (recommended)
- `ml_decoder_phase3.pth` - Improved training model
- `ml_decoder_phase4_deep.pth` - Deep architecture
- `ml_decoder_phase4_wide.pth` - Wide architecture
- `ml_decoder_phase4_residual.pth` - Residual architecture (best)

### Configuration
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

## Archived Files (`archive/`)

Old/redundant files have been moved to archive:
- `old_files/` - Old scripts (hard-decision training, verification scripts, examples)
- `old_models/` - Old model files (hard-decision model)
- `old_data/` - Old/redundant data files and plots

## Quick Start

1. **Setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train Model**:
   ```bash
   python train_ml_decoder_soft.py --num_samples 100000 --epochs 50
   ```

3. **Evaluate**:
   ```bash
   python final_evaluation_phase5.py
   ```

4. **Compare**:
   ```bash
   python compare_decoders.py --model_path models/ml_decoder_direct_soft.pth --use_soft_input
   ```

## File Organization

```
ECE432FinalProject/
├── src/                    # Core source code (ESSENTIAL)
├── tests/                  # Unit tests (ESSENTIAL)
├── models/                 # Trained ML models
├── data/                   # Results and plots
├── docs/                   # Documentation
├── archive/                # Archived old files
│   ├── old_files/         # Old scripts
│   ├── old_models/       # Old models
│   └── old_data/         # Old data files
├── train_ml_decoder_soft.py      # Main training script
├── train_ml_decoder_phase3.py   # Improved training
├── train_ml_decoder_phase4.py   # Architecture variants
├── run_baseline_evaluation.py   # Baseline evaluation
├── compare_decoders.py           # Two-model comparison
├── final_evaluation_phase5.py   # Comprehensive evaluation
├── README.md                    # Main documentation
├── PROJECT_EXPLANATION.txt      # Project explanation
├── PROJECT_SUMMARY.md           # Project summary
├── VERIFICATION_REPORT.txt      # Verification report
├── requirements.txt             # Dependencies
└── .gitignore                   # Git ignore rules
```

