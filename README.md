# ML-Assisted Hamming Code Decoder

This project implements a machine learning-assisted Hamming(7,4) decoder that improves upon classical decoding performance over AWGN channels. The project explores whether neural networks can enhance traditional hard-decision decoding by learning complex error patterns.

## Project Overview

The Hamming(7,4) code encodes 4 data bits into 7 codeword bits, allowing detection and correction of single-bit errors. This project:
- Implements classical syndrome-based decoding
- Develops ML-assisted decoders using neural networks
- Compares performance over AWGN channels
- Evaluates Bit Error Rate (BER) vs. Eb/N0 curves

## Project Structure

```
ECE432FinalProject/
├── src/                    # Main source code
│   ├── hamming.py          # Hamming(7,4) encoder/decoder
│   ├── classical_decoder.py # Classical syndrome decoder
│   ├── channel.py          # BPSK modulation & AWGN channel
│   ├── evaluation.py       # BER evaluation framework
│   ├── ml_decoder.py       # ML model architectures
│   └── ml_evaluation.py    # ML decoder evaluation
├── tests/                  # Unit tests (66 tests total)
├── notebooks/              # Analysis notebooks
├── models/                 # Saved ML models
├── data/                   # Results and plots
├── verify_phase*.py        # Phase verification scripts
├── train_ml_decoder.py     # ML model training script
├── compare_decoders.py     # Comparison script
└── generate_final_results.py # Final results generation
```

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

### Phase 5: ML-Assisted Decoding Integration ✓ COMPLETE
- ML decoder integrated into evaluation pipeline
- ML decoder performance evaluation framework
- Comparison functionality for classical vs. ML
- Comparison plotting and analysis
- All unit tests passing (5/5)

### Phase 6: Final Comparison & Documentation ✓ COMPLETE
- Final results generation script (`generate_final_results.py`)
- Comprehensive comparison plots and analysis
- Complete README documentation with full workflow
- Reproducibility ensured with fixed seeds
- Example usage scripts (`example_usage.py`)
- Project summary document
- All verification scripts present and working

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

# Verify Phase 5: ML-assisted decoding integration
python verify_phase5.py

# Compare classical and ML decoders
python compare_decoders.py --model_path models/ml_decoder_direct.pth --approach direct

# Generate final results and comparison
python generate_final_results.py --model_path models/ml_decoder_direct.pth --approach direct
```

### Run Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Complete Workflow

### 1. Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Implementation
```bash
# Verify each phase
python verify_phase1.py  # Hamming code
python verify_phase2.py  # Channel simulation
python verify_phase3.py  # Baseline evaluation
python verify_phase4.py  # ML models
python verify_phase5.py  # ML integration
```

### 3. Generate Baseline Results
```bash
# Run classical decoder evaluation
python run_baseline_evaluation.py
# Results saved to: data/baseline_ber_curve.png
```

### 4. Train ML Model
```bash
# Train direct mapping decoder
python train_ml_decoder.py \
    --approach direct \
    --num_samples 100000 \
    --epochs 50 \
    --ebno_min -5.0 \
    --ebno_max 10.0 \
    --output models/ml_decoder_direct.pth

# Train post-processing decoder
python train_ml_decoder.py \
    --approach post \
    --num_samples 100000 \
    --epochs 50 \
    --output models/ml_decoder_post.pth
```

### 5. Compare Decoders
```bash
# Compare classical vs. ML decoder
python compare_decoders.py \
    --model_path models/ml_decoder_direct.pth \
    --approach direct \
    --num_bits 100000

# Generate final results
python generate_final_results.py \
    --model_path models/ml_decoder_direct.pth \
    --approach direct \
    --num_bits 100000
```

## Reproducibility

All simulations use fixed random seeds for reproducibility:
- Default seed: 42
- Can be specified via command-line arguments
- Results are deterministic with same seed

To reproduce results:
1. Use the same random seed (default: 42)
2. Use the same number of bits per evaluation point
3. Use the same Eb/N0 range and number of points

## Key Results

The project demonstrates:
- **Classical Decoder**: Baseline Hamming(7,4) performance with hard-decision decoding
- **ML-Assisted Decoder**: Potential improvement through learned error correction
- **Comparison**: Side-by-side BER vs. Eb/N0 curves showing relative performance

## Technical Details

### Hamming(7,4) Code
- **Code Rate**: 4/7 (4 data bits → 7 codeword bits)
- **Error Correction**: Single-bit errors
- **Generator Matrix**: Systematic form with parity equations
- **Syndrome Decoding**: Standard lookup table approach

### Channel Model
- **Modulation**: BPSK (0 → +1, 1 → -1)
- **Channel**: AWGN (Additive White Gaussian Noise)
- **Demodulation**: Hard-decision (threshold at 0)

### ML Architecture
- **Direct Mapping**: 7 received bits → 4 data bits
- **Post-Processing**: 7 classical decoder bits → 4 corrected data bits
- **Network**: Fully connected layers with ReLU and Dropout
- **Output**: Sigmoid activation for binary classification

## Dependencies

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- PyTorch >= 1.12.0
- Matplotlib >= 3.5.0
- pytest >= 7.0.0

## Testing

All 66 unit tests pass:
- 15 tests: Hamming encoder/decoder
- 24 tests: Channel simulation
- 11 tests: Evaluation framework
- 11 tests: ML decoder
- 5 tests: ML evaluation

Run tests: `pytest tests/ -v`

## Authors

Shashwat Sinha, Ambarish Pathak

ECE 432 - November 2025

