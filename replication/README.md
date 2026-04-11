# Replication Materials for fastLPR JSS Submission

This directory contains materials to replicate all results and figures from the fastLPR Journal of Statistical Software (JSS) submission.

---

## Contents

- `README.md` - This file (replication instructions)
- `reproduce_paper.sh` - Bash script to reproduce all figures
- `reproduce_paper.bat` - Windows batch script (legacy; may not work in all environments)

---

## System Requirements

### Software Requirements

- **Python:** 3.9 or later (tested on 3.9, 3.10, 3.11, 3.12)
- **Operating System:** Windows, Linux, or macOS
- **Memory:** 4GB RAM minimum (8GB recommended for 3D examples)
- **Disk Space:** 500MB for package and data

### Package Dependencies

Required packages:
- `numpy >= 1.23`
- `scipy >= 1.10`
- `matplotlib >= 3.7`

---

## Installation Instructions

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository (or unzip the submission archive)
cd fastLPR_py

# Create a virtual environment (recommended)
uv venv

# Install the package from source
uv pip install -e .

# Verify installation
uv run python -c "from fastlpr import cv_fastlpr; print('fastLPR installed successfully')"
```

### Option 2: Install from PyPI

```bash
pip install fastlpr
```

---

## Reproducing All Figures

### Automated Reproduction (Recommended)

**On Linux/macOS:**
```bash
cd fastLPR_py
bash ../replication/reproduce_paper.sh
```

**On Windows (Git Bash recommended):**
```bash
cd fastLPR_py
bash ../replication/reproduce_paper.sh
```

This will:
1. Run all 5 figure generation scripts (Fig 2-6)
2. Save figures to `fastLPR_py/fig/reproduced/`
3. Generate a summary report
4. Display timing information for each figure

**Expected Runtime:**
- Figure 2 (KDE 1D/2D/3D): ~50 seconds
- Figure 3 (Boundary comparison): ~1 second
- Figure 4 (Complex regression): ~9 seconds
- Figure 5 (Heteroscedastic): ~16 seconds
- Figure 6 (Applications): ~72 seconds
- **Total: ~150 seconds (2.5 minutes)**

### Manual Reproduction (Individual Figures)

To reproduce individual figures:

```bash
cd fastLPR_py/examples

# Figure 2: Kernel Density Estimation (1D, 2D, 3D)
python example_fig2_fastkde.py

# Figure 3: Boundary Comparison (NW vs LL vs LQ)
python example_fig3_boundary_comparison.py

# Figure 4: Complex-Valued Regression
python example_fig4_complex.py

# Figure 5: Heteroscedastic Regression (1D and 2D)
python example_fig5_heteroscedasticity.py

# Figure 6: Real-World Applications (qEEG and MRI)
python example_fig6_applications.py
```

**Output Location:** All figures are saved to `fastLPR_py/fig/reproduced/` in both PNG (300 DPI) and PDF formats.

---

## Verification

### Verifying Correctness

All examples use fixed random seeds for reproducibility. To verify correctness against MATLAB reference:

```bash
cd fastLPR_py

# Run cross-language verification tests
pytest tests/xl/

# Expected output:
# - Maximum absolute error < 1% for all examples
# - Bandwidth selection matches MATLAB exactly
```

### Numerical Comparison
```bash
# Run comprehensive cross-validation tests
cd fastLPR_py
pytest tests/xl/ -v

# Expected results:
# - Fig2 (KDE): < 1% error
# - Fig3 (Boundary): < 1% error
# - Fig4 (Complex): 0.12% error (exact bandwidth match)
# - Fig5 (Heteroscedastic): 0.25% error
# - Fig6 (Applications): Verified against MATLAB
```

---

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'fastlpr'**
- Solution: Install the package using `pip install -e .` from `fastLPR_py/` directory
- Verify: `python -c "import fastlpr"`

**2. ModuleNotFoundError: scipy, numpy, matplotlib**
- Solution: Install dependencies with `uv pip install -e .`
- Note: Requires numpy >= 1.23, scipy >= 1.10, matplotlib >= 3.7

**3. Figures look different across environments**
- Cause: Different matplotlib, numpy, or platform versions
- Solution: Re-run with the same package environment and compare numerical test output
- Note: Minor visual differences are acceptable if numerical results match

**4. 3D KDE (Fig2c, Fig2f) runs very slowly**
- Expected: Python is 13× slower than MATLAB for 3D KDE (49.7s vs 3.7s)
- Reason: Interpreter overhead, not algorithm issue
- Note: Computation is CORRECT, just slower
- Workaround: Use quick test mode: `export FASTLPR_QUICK_TEST=1` (50% sample size)

**5. Memory error on Fig6 (MRI data)**
- Cause: Large dataset (256×256 MRI image)
- Solution: Requires 4GB+ RAM
- Workaround: Reduce grid size in example_fig6_applications.py

### Platform-Specific Notes

**Windows:**
- Use Git Bash and run the Bash script: `bash ../replication/reproduce_paper.sh`
- Use `uv` to manage/install dependencies

**Linux/macOS:**
- Use forward slashes in paths: `cd fastLPR_py/examples`
- May use `python3` instead of `python`
- Activate venv: `source .venv/bin/activate`

**macOS Apple Silicon (M1/M2):**
- All dependencies support ARM64 natively
- No special installation needed
- Performance similar to Intel

---

## Quick Test Mode

For faster testing (50% sample sizes, reduced grid resolution):

**Linux/macOS:**
```bash
export FASTLPR_QUICK_TEST=1
python examples/reproduce_all_figures.py
```

**Windows:**
```cmd
set FASTLPR_QUICK_TEST=1
python examples\reproduce_all_figures.py
```

**Quick Test Runtime:** ~75 seconds (50% of normal)

**Note:** Quick test mode is for development/testing only. For publication-quality figures, use normal mode.

---

## Testing Against MATLAB Reference

To verify exact numerical agreement with MATLAB implementation:

```bash
cd fastLPR_py

# Install test dependencies
uv pip install pytest pytest-cov

# Run cross-language verification tests
pytest tests/xl/ -v

# Expected output:
# test_complex_regression_vs_matlab PASSED
# test_1d_heteroscedastic_vs_matlab PASSED
# test_2d_heteroscedastic_vs_matlab PASSED
```

**Verification Criteria:**
- Maximum absolute error < 1% for all examples
- Bandwidth selection matches MATLAB exactly
- Grid values match within numerical precision

---

## Performance Benchmarks

Typical execution times on modern hardware (Intel i7/AMD Ryzen, 16GB RAM):

| Figure | Description | Python Time | MATLAB Time | Ratio |
|--------|-------------|-------------|-------------|-------|
| Fig2 | KDE (1D/2D/3D) | 50.9s | ~30s | 1.7× |
| Fig3 | Boundary | 1.0s | ~0.8s | 1.3× |
| Fig4 | Complex | 9.1s | ~7s | 1.3× |
| Fig5 | Heteroscedastic | 15.6s | ~12s | 1.3× |
| Fig6 | Applications | 72.3s | ~50s | 1.4× |
| **Total** | All figures | **~149s** | **~100s** | **1.5×** |

**Note:** Python is 1.3-1.7× slower than MATLAB (13× for 3D KDE only). This is expected due to interpreter overhead, not algorithm differences. The O(N + M log M) complexity is preserved.

---

## Documentation

For detailed documentation, see:
- **User Guide:** `fastLPR_py/README.md`
- **Installation Guide:** `fastLPR_py/INSTALL.md` (877 lines, comprehensive)
- **API Documentation:** Function docstrings in `src/fastlpr/`
- **Changelog:** `fastLPR_py/CHANGELOG.md`
- **Testing Guide:** `fastLPR_py/tests/TEST_SUMMARY.md`

---

## Citation

If you use fastLPR in your research, please cite:

```bibtex
@article{WangLiPazLinaresValdesSosa2026,
  title  = {{fastLPR}: Fast Local Polynomial Regression via {NUFFT}
            in {MATLAB}, {Python}, and {R}},
  author = {Wang, Ying and Li, Min and Paz-Linares, Deirel
            and Valdes-Sosa, Pedro A.},
  journal = {Submitted to Journal of Statistical Software},
  year    = {2026}
}
```

---

## Support

**Bug Reports:** Please report issues on GitHub or contact the authors directly.

**Authors:**
- Ying Wang: yingwangrigel@gmail.com
- Min Li: minli.231314@gmail.com

---

## License

GNU General Public License v3.0 or later (GPL-3.0+)

Copyright (c) 2019-2026 Ying Wang, Min Li, Deirel Paz-Linares, Pedro A. Valdes-Sosa

---

**Document Version:** 1.0.0
**Last Updated:** 2026-04-11
**Replication Archive:** JSS Submission
