# ML-Assisted Hamming Code Decoder

This project implements a machine learning-assisted Hamming(7,4) decoder that improves upon classical decoding performance over AWGN channels.

## Project Structure

- `src/` - Main source code
- `tests/` - Unit tests
- `notebooks/` - Analysis notebooks
- `models/` - Saved ML models
- `data/` - Data files

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
pytest tests/
```

## Project Status

### Phase 1: Foundation & Classical Decoder ✓ COMPLETE
- Hamming(7,4) encoder implemented
- Classical syndrome-based decoder implemented
- All unit tests passing (15/15)
- Verified: All 16 possible 4-bit messages encode/decode correctly
- Verified: All single-bit errors are detected and corrected

### Phase 2: Channel Simulation ✓ COMPLETE
- BPSK modulation implemented (0 → +1, 1 → -1)
- AWGN channel with correct noise statistics
- Hard-decision and soft-decision (LLR) demodulation
- Eb/N0 to noise variance conversion
- Complete transmission simulation pipeline
- All unit tests passing (24/24)

### Phase 3: Baseline Performance Evaluation ✓ COMPLETE
- BER calculation framework implemented
- Classical decoder performance evaluation across Eb/N0 range
- BER vs. Eb/N0 plotting functionality
- Complete evaluation pipeline with reproducibility
- All unit tests passing (11/11)

### Phase 4: ML Model Development ✓ COMPLETE
- Neural network architectures implemented (DirectMapping and PostProcessing)
- Training data generation pipeline
- Model training framework with validation
- Model saving/loading functionality
- MLDecoder wrapper class for inference
- All unit tests passing (11/11)

### Phase 5: ML-Assisted Decoding Integration (Next)
- Integrate ML model with decoding pipeline
- Evaluate ML-assisted decoder performance
- Compare with classical baseline

## Usage

### Verify Phases
```bash
# Verify Phase 1: Hamming encoder/decoder
python verify_phase1.py

# Verify Phase 2: Channel simulation
python verify_phase2.py

# Verify Phase 3: Baseline evaluation
python verify_phase3.py

# Run full baseline evaluation (takes longer, better statistics)
python run_baseline_evaluation.py

# Verify Phase 4: ML model development
python verify_phase4.py

# Train ML decoder model
python train_ml_decoder.py --approach direct --num_samples 100000 --epochs 50
```

### Run Tests
```bash
pytest tests/
```

## Authors

Shashwat Sinha, Ambarish Pathak

