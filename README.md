# FFPR: Fractal Floating-Point Representation

**FFPR** is an engineering extension of the IEEE 754 standard, designed for deterministic handling of singularities, including `NaN` and `Infinity`, in deep learning architectures.

## Overview
Standard floating-point arithmetic frequently encounters numerical instability in deep learning, leading to `NaN` values, vanishing gradients, or exploding outputs. **Fractal Floating-Point Representation (FFPR)** introduces a novel algebraic framework that maps these singularities into a continuum of ordered generators, `Fk`, allowing for robust and deterministic computation.

### Key Features
- **Singularity Management:** Deterministic resolution of `0/0`, `∞/∞`, and various power-based indeterminacies in constant time.
- **Threshold Promotion:** An automated mechanism to prevent gradient underflow and overflow before they trigger hardware-level errors.
- **Memory Efficiency:** Implements a **Sparse Fractal State** that preserves numerical history while consuming minimal GPU memory (typically 1–5% overhead).
- **PyTorch Native:** Designed for integration into modern training loops without requiring custom hardware.

## Repository Contents
- `FFPR_Preprint.pdf`: The complete theoretical framework and mathematical derivation
- `Stress_test.py`: Source code and empirical results from stress-testing environments

## Current Status
- **Theory**: The mathematical foundation is fully established and detailed in the preprint.
- **Implementation**: Empirical validation shows that FFPR maintains computational stability in environments where standard IEEE 754 training regimes collapse
MIT.license
