# Project Organization

This project has been organized to keep only essential files in the main directory.

## Quick File Guide

### Essential Scripts (Root Directory)
- `train_ml_decoder_soft.py` - Train soft-decision ML decoder (RECOMMENDED)
- `train_ml_decoder_phase3.py` - Train with improved training strategy
- `train_ml_decoder_phase4.py` - Train architecture variants
- `run_baseline_evaluation.py` - Evaluate classical decoder
- `compare_decoders.py` - Compare two decoders
- `final_evaluation_phase5.py` - Comprehensive evaluation (all models)

### Core Code
- `src/` - All source code modules

### Tests
- `tests/` - All unit tests (66 tests)

### Models
- `models/` - Trained ML models (5 models)

### Results
- `data/` - Final results and plots

### Documentation
- `README.md` - Main documentation
- `PROJECT_EXPLANATION.txt` - Complete project explanation
- `PROJECT_SUMMARY.md` - Project summary
- `VERIFICATION_REPORT.txt` - Results verification
- `docs/` - Detailed technical documentation

### Archived Files
- `archive/` - Old/redundant files moved here
  - `old_files/` - Old scripts (verification, examples, hard-decision training)
  - `old_models/` - Old model files
  - `old_data/` - Old data files and plots

## What Was Removed/Archived

1. **Old Training Scripts**: Hard-decision training (replaced by soft-decision)
2. **Verification Scripts**: Phase verification scripts (not needed for final)
3. **Example Scripts**: Example usage (not essential)
4. **Old Models**: Hard-decision model (not used)
5. **Old Data Files**: Redundant plots and old results
6. **Empty Directories**: notebooks/, docs/scripts/

## Current Structure

```
ECE432FinalProject/
├── src/                    # Core code (6 modules)
├── tests/                  # Tests (5 test files)
├── models/                 # Trained models (5 models)
├── data/                   # Results (8 files)
├── docs/                   # Documentation (5 files)
├── archive/                # Archived old files
├── [6 training/evaluation scripts]
├── [5 documentation files]
├── requirements.txt
└── .gitignore
```

## File Count

- **Root scripts**: 6 essential Python scripts
- **Data files**: 8 essential files (plots, results, reports)
- **Models**: 5 trained models
- **Documentation**: 5 main docs + 5 detailed docs

Total essential files: ~30 files (excluding venv, __pycache__)

## Usage

See `README.md` for complete usage instructions.

For project explanation, see `PROJECT_EXPLANATION.txt`.

For verification of results, see `VERIFICATION_REPORT.txt`.

