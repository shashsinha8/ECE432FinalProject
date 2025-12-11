# ML-Assisted Hamming Code Decoder

This project implements a machine learning-assisted Hamming(7,4) decoder that improves upon classical decoding performance over AWGN channels. The project demonstrates that neural networks can significantly outperform classical decoders by learning complex error patterns.

## Quick Start

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train Model
```bash
# Train soft-decision ML decoder (recommended)
python scripts/training/train_ml_decoder_soft.py --num_samples 100000 --epochs 50

# Train with improved training strategy
python scripts/training/train_ml_decoder_phase3.py --use_codeword_loss --augment_ratio 0.1

# Train architecture variants
python scripts/training/train_ml_decoder_phase4.py --architecture residual
```

### Evaluate
```bash
# Run comprehensive evaluation (all models)
python scripts/evaluation/final_evaluation_phase5.py

# Compare two decoders
python scripts/evaluation/compare_decoders.py --model_path models/ml_decoder_direct_soft.pth --use_soft_input

# Run baseline evaluation
python scripts/evaluation/run_baseline_evaluation.py
```

### Run Tests
```bash
pytest tests/
```

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
│   ├── main/              # Main documentation files
│   └── [detailed docs]   # Detailed technical docs
├── archive/                # Archived old files
└── README.md              # This file
```

## Key Results

- **Soft-Decision ML Decoder**: 43.3% average improvement over classical decoder
- **Best Architecture**: Residual architecture (44.5% improvement)
- **Critical Innovation**: Using Log-Likelihood Ratios (LLRs) instead of hard bits

## Documentation

- **Quick Start**: `docs/main/QUICK_START.md`
- **Complete Explanation**: `docs/main/PROJECT_EXPLANATION.txt`
- **Project Summary**: `docs/main/PROJECT_SUMMARY.md`
- **Verification Report**: `docs/main/VERIFICATION_REPORT.txt`
- **Technical Details**: `docs/CODEBASE_DOCUMENTATION.md`

## Results

- **Final Comparison**: `data/phase5_final_comparison.png`
- **Performance Report**: `data/phase5_final_report.md`
- **Raw Results**: `data/phase5_final_results.npy`

## Authors

Shashwat Sinha, Ambarish Pathak  
ECE 432 - December 2025
