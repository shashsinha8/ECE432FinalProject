# Quick Start Guide

## Project Structure

```
ECE432FinalProject/
├── src/                    # Core source code
├── tests/                  # Unit tests
├── scripts/                # Executable scripts
│   ├── training/          # Training scripts
│   └── evaluation/        # Evaluation scripts
├── models/                 # Trained ML models
├── data/                   # Results and plots
├── docs/                   # Documentation
│   ├── main/              # Main documentation
│   └── [technical docs]  # Detailed technical docs
└── README.md              # Main README
```

## Essential Commands

### Training
```bash
# Train soft-decision ML decoder (recommended)
python scripts/training/train_ml_decoder_soft.py --num_samples 100000 --epochs 50

# Train with improved training
python scripts/training/train_ml_decoder_phase3.py --use_codeword_loss

# Train architecture variants
python scripts/training/train_ml_decoder_phase4.py --architecture residual
```

### Evaluation
```bash
# Comprehensive evaluation (all models)
python scripts/evaluation/final_evaluation_phase5.py

# Compare two decoders
python scripts/evaluation/compare_decoders.py --model_path models/ml_decoder_direct_soft.pth --use_soft_input

# Baseline evaluation
python scripts/evaluation/run_baseline_evaluation.py
```

## Documentation

- **Main README**: `README.md`
- **Complete Explanation**: `docs/main/PROJECT_EXPLANATION.txt`
- **Project Summary**: `docs/main/PROJECT_SUMMARY.md`
- **Verification**: `docs/main/VERIFICATION_REPORT.txt`

## Results

- **Final Comparison Plot**: `data/phase5_final_comparison.png`
- **Performance Report**: `data/phase5_final_report.md`
- **Raw Results**: `data/phase5_final_results.npy`

