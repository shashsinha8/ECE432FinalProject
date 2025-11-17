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

### Phase 2: Channel Simulation (Next)
- BPSK modulation
- AWGN channel
- Demodulation

## Usage

### Verify Phase 1
```bash
python verify_phase1.py
```

### Run Tests
```bash
pytest tests/
```

## Authors

Shashwat Sinha, Ambarish Pathak

