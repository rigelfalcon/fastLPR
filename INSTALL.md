# Installation Guide - fastLPR Project

This guide provides comprehensive installation instructions for all three implementations of the fastLPR (Fast Local Polynomial Regression) toolbox.

**Choose your implementation:**
- [Python](#python-installation) - Recommended for most users
- [MATLAB](#matlab-installation) - Reference implementation
- [R](#r-installation) - Limited functionality (1D/2D only, orders 0/1/2)

---

## Python Installation

### Requirements

**Python Version:**
- Python 3.9 or later
- Tested on: Python 3.9, 3.10, 3.11, 3.12, 3.13

**Required Dependencies:**
- numpy >= 1.23.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0

**Optional Dependencies:**
- numba >= 0.57.0 - JIT compilation for 10-20% speedup
- pytest >= 7.4 - For running tests
- black >= 23.7 - Code formatting
- ruff >= 0.1.6 - Linting

### Installation Methods

#### Method 1: From Source (Recommended)

This method is recommended if you want to:
- Run examples and reproduce paper figures
- Modify the code or contribute
- Access verification scripts

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/fastLPR.git
cd fastLPR/fastLPR_py

# 2. Create virtual environment (recommended)
python -m venv .venv

# 3. Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Windows CMD:
.venv\Scripts\activate.bat

# 4. Install in editable mode
pip install -e .

# 5. Verify installation
python -c "from fastlpr import cv_fastlpr; print('fastLPR installed successfully!')"
```

#### Method 2: Direct Install from Git

```bash
# Install directly from repository
pip install -e "git+https://github.com/your-repo/fastLPR.git#egg=fastlpr&subdirectory=fastLPR_py"
```

#### Method 3: Development Installation

For developers who want to run tests and use code quality tools:

```bash
cd fastLPR/fastLPR_py
pip install -e ".[dev]"
```

This installs additional tools: pytest, pytest-cov, black, ruff, mypy.

### Verification

#### Quick Test

Run this minimal test to verify the installation:

```python
import numpy as np
from fastlpr import cv_fastlpr, get_hlist

# Generate test data
np.random.seed(42)
x = np.random.rand(100, 1)
y = x**2 + 0.1*np.random.randn(100, 1)

# Run regression with bandwidth selection
hlist = get_hlist(10, [0.01, 1.0])
opt = {'order': 1, 'dstd': 0}
regs = cv_fastlpr(x, y, hlist, opt)

print(f"✓ fastLPR is working! Selected bandwidth: {regs.bandwidth}")
```

Expected output:
```
✓ fastLPR is working! Selected bandwidth: [0.07742637]
```

#### Run Examples

Test the examples to ensure everything works:

```bash
cd examples

# Run individual examples
python example_fig2_fastkde.py
python example_fig3_boundary_comparison.py
python example_fig4_complex.py
python example_fig5_heteroscedasticity.py
python example_fig6_applications.py

# Or run all examples at once
python reproduce_all_figures.py
```

#### Run Test Suite

For comprehensive verification:

```bash
cd tests
pytest test_matlab_python_cross_validation.py -v
```

Expected output: All tests should pass.

---

## MATLAB Installation

### Requirements

**MATLAB Version:**
- MATLAB R2020a or later
- Recommended: MATLAB R2021a or later
- **No toolboxes required** - uses base MATLAB only

### Installation Steps

#### Step 1: Add to MATLAB Path

```matlab
% Navigate to fastLPR directory
cd path/to/fastLPR

% Run setup script (adds folders to path)
fastLPR_setup();
```

The `fastLPR_setup()` function automatically adds:
- `utility/` - Public API functions
- `utility/core/` - Internal functions
- `example/` - Example scripts

#### Step 2: Verify Installation

```matlab
% Check that main functions are accessible
help cv_fastLPR
help cv_fastKDE
help fastLPR_predict

% Run a simple test
x = rand(500, 1);
y = sin(2*pi*x) + 0.1*randn(500, 1);
hlist = get_hList(20, [0.01, 1], @logspace);
opt.order = 1;
regs = cv_fastLPR(x, y, hlist, opt);
fastLPR_plot(regs.fpp_yhat);
```

#### Step 3: Run Examples

```matlab
% Navigate to examples directory
cd example

% Run individual examples
example_fig2_fastkde           % 1D and 2D KDE
example_fig3_boundary_comparison % NW vs LL vs LQ
example_fig4_complex           % Complex-valued data
example_fig5_heteroscedasticity % Variance estimation
example_fig6_applications      % Real-world data

% Or run all examples
reproduce_all_figures          % Generates all paper figures
```

#### Step 4: Run Test Suite

```matlab
cd tests
run_all  % Runs all 12 tests
```

Expected output: `All 12 tests passed!`

### Persistent Setup (Optional)

To avoid running `fastLPR_setup()` every session, add to your `startup.m`:

```matlab
% Edit MATLAB startup file
edit(fullfile(userpath, 'startup.m'))

% Add this line:
run('path/to/fastLPR/fastLPR_setup.m');
```

---

## R Installation

### Requirements

**R Version:**
- R >= 4.0.0
- Tested on: R 4.0, 4.1, 4.2, 4.3, 4.4, 4.5

**Required Packages:**
- stats (base)
- utils (base)
- Matrix >= 1.0.0
- fftw >= 2.0.0
- interp >= 1.0.0
- akima >= 0.6.0

**Suggested Packages:**
- testthat >= 3.0.0 (testing)
- roxygen2 >= 7.0.0 (documentation)
- ggplot2 (enhanced plotting)

### Important Limitations

The R implementation is ~95% complete with the following status:
- ✅ **Working**: 1D/2D regression, orders 0/1/2, KDE, GCV bandwidth selection
- ⚠️ **Limited**: 2D variance estimation, 3D support not implemented
- ✅ **Verified**: Cross-validated against MATLAB (different RNGs explain bandwidth differences)

### Installation Steps

#### Step 1: Install R

Download and install R from [CRAN](https://cran.r-project.org/):
- Windows: Use the installer
- macOS: Use the .pkg file
- Linux: Use your package manager (`sudo apt-get install r-base`)

#### Step 2: Install Dependencies

```r
# Install required packages from CRAN
install.packages(c("Matrix", "fftw", "interp", "akima"))

# Optional: Install suggested packages
install.packages(c("testthat", "roxygen2", "ggplot2"))
```

#### Step 3: Source Setup Script

```r
# Navigate to fastLPR_R directory
setwd("path/to/fastLPR_R")

# Source setup script
source("setup.R")
```

The `setup.R` script:
- Loads all required packages
- Sources all R function files
- Checks for missing dependencies

#### Step 4: Verify Installation

```r
# Test 1D KDE
x <- matrix(rnorm(200), ncol=1)
hlist <- get_hlist(10, c(0.1, 1.0))
kde <- cv_fastkde(x, hlist)
print(paste("Selected bandwidth:", kde$h_opt))

# Test 1D regression (order 1)
x <- matrix(runif(100), ncol=1)
y <- sin(2*pi*x) + rnorm(100, sd=0.1)
hlist <- get_hlist(10, c(0.01, 1.0))
opt <- list(order=1, N=100, dstd=0)
regs <- cv_fastlpr(x, y, hlist, opt)
print(paste("Selected bandwidth:", regs$h_opt))
```

#### Step 5: Run Examples

```r
# Navigate to examples directory
setwd("examples")

# Run basic examples
source("example_kde_1d.R")
source("example_regression_1d.R")
```

### Using reticulate (Alternative)

For full functionality, use the Python implementation via `reticulate`:

```r
# Install reticulate
install.packages("reticulate")

# Configure Python
library(reticulate)
use_python("/path/to/python")  # or use_virtualenv(".venv")

# Import fastlpr
fastlpr <- import("fastlpr")

# Use Python functions from R
x <- r_to_py(matrix(rnorm(200), ncol=1))
hlist <- r_to_py(c(0.01, 0.05, 0.1, 0.5, 1.0))
kde <- fastlpr$cv_fastkde(x, hlist)
```

---

## Troubleshooting

### Python Issues

#### Issue: `ModuleNotFoundError: No module named 'fastlpr'`

**Cause:** Package not installed or wrong environment active.

**Solutions:**
1. Ensure virtual environment is activated:
   ```bash
   # Check current environment
   which python  # Linux/macOS
   where python  # Windows
   ```
2. Install in editable mode from correct directory:
   ```bash
   cd fastLPR_py
   pip install -e .
   ```
3. Verify installation:
   ```bash
   pip list | grep fastlpr
   ```

#### Issue: `ImportError: numpy.core.multiarray failed to import`

**Cause:** NumPy version incompatibility.

**Solution:**
```bash
pip install --upgrade numpy>=1.23.0 scipy>=1.10.0
```

#### Issue: Figures don't match MATLAB exactly

**Cause:** Different random number generators (expected behavior).

**Solution:** This is **not an error**. Python and MATLAB use different RNGs, so:
- Bandwidth selection may differ slightly (both are valid)
- Visual appearance differs, but numerical accuracy is preserved (< 1e-3 MSE)
- Focus on numerical accuracy, not visual match

#### Issue: Slow performance on AMD CPUs

**Cause:** NumPy built with Intel MKL (optimized for Intel CPUs only).

**Solution - Use OpenBLAS:**
```bash
# Option 1: Using conda (recommended for AMD)
conda install numpy "libblas=*=*openblas"

# Option 2: Using pip
pip uninstall numpy
pip install numpy openblas

# Verify (should show openblas)
python -c "import numpy as np; np.show_config()"
```

Expected speedup: 1.5-3x on AMD Ryzen CPUs.

#### Issue: Out of memory on large datasets

**Cause:** Grid size too large for available RAM.

**Solutions:**
1. Reduce grid size in options:
   ```python
   opt = {'order': 1, 'N': 64}  # Default is 100
   ```
2. Use fewer bandwidth candidates:
   ```python
   hlist = get_hlist(10, [0.01, 1.0])  # Instead of 20
   ```
3. Process dimensions separately (for multi-dimensional data)

#### Issue: `matplotlib` not found when running examples

**Cause:** matplotlib not installed.

**Solution:**
```bash
pip install matplotlib>=3.7.0
```

#### Issue: Examples fail with `FileNotFoundError` for data files

**Cause:** Running from wrong directory.

**Solution:**
```bash
# Always run examples from the examples directory
cd fastLPR_py/examples
python example_fig2_fastkde.py
```

#### Issue: Tests fail on Windows

**Cause:** Path separator differences or line ending issues.

**Solution:**
```bash
# Convert line endings (if cloned without git autocrlf)
pip install dos2unix
dos2unix src/fastlpr/*.py

# Or use WSL (Windows Subsystem for Linux)
wsl
cd /mnt/c/path/to/fastLPR_py
pip install -e .
```

### MATLAB Issues

#### Issue: `Undefined function or variable 'cv_fastLPR'`

**Cause:** fastLPR not on MATLAB path.

**Solution:**
```matlab
% Navigate to fastLPR directory
cd path/to/fastLPR

% Run setup
fastLPR_setup();

% Verify
which cv_fastLPR  % Should show path to function
```

#### Issue: `Error using private/nufftn_type1`

**Cause:** Calling private function from wrong location.

**Solution:** Private functions in `utility/core/` can only be called by functions in `utility/`. Do not call them directly from scripts or command window.

#### Issue: Very slow performance (>10 seconds for small data)

**Cause:** Not using vectorization or hitting interpreted bottleneck.

**Solution:**
1. Ensure you're not using nested loops in custom code
2. Check MATLAB version (older versions may be slower)
3. For small datasets (N < 2000), consider naive methods may be faster
4. Profile the code to find bottlenecks:
   ```matlab
   profile on
   regs = cv_fastLPR(x, y, hlist, opt);
   profile viewer
   ```

#### Issue: Figures missing or blank after running examples

**Cause:** Subplot positioning issues (known bug in Figure 3).

**Solution:** This is a known issue documented in TODO.md. The main plot disappears when insets are created. Workaround: Save figures before insets are added, or adjust subplot positioning manually.

#### Issue: Out of memory on large datasets

**Cause:** Grid size or bandwidth list too large.

**Solutions:**
1. Reduce grid size:
   ```matlab
   opt.N = 64;  % Default is 100
   ```
2. Use fewer bandwidth candidates:
   ```matlab
   hlist = get_hList(10, [0.01, 1], @logspace);  % Instead of 20
   ```
3. Process dimensions separately
4. Use parallel computing (if Parallel Computing Toolbox available):
   ```matlab
   opt.use_parallel = true;
   ```

### R Issues

#### Issue: `Error: package 'Matrix' is not installed`

**Cause:** Missing required packages.

**Solution:**
```r
# Install all required packages
install.packages(c("Matrix", "fftw", "interp", "akima"))

# Retry setup
source("setup.R")
```

#### Issue: `Error in cv_fastlpr(): Order 2 not implemented`

**Cause:** You're trying functionality that isn't implemented yet.

**Solution:** Use orders 0, 1, or 2 for 1D/2D data. These are fully implemented and verified.

#### Issue: Different results from MATLAB/Python

**Cause:** Different random number generators (expected).

**Solution:** This is normal. Each language uses different RNGs, resulting in:
- Different bandwidth selection (both valid)
- Slight numerical differences in stochastic estimates
- Focus on algorithm correctness, not exact numeric match

**To verify correctness:** Compare with fixed input data (not random):
```r
# Use deterministic data
x <- matrix(seq(0, 1, length.out=100), ncol=1)
y <- sin(2*pi*x)
```

#### Issue: Slow performance compared to MATLAB

**Cause:** R is interpreted, MATLAB has JIT compilation.

**Solutions:**
1. Use larger datasets where NUFFT overhead is amortized
2. Consider using reticulate to call Python implementation
3. For small datasets (N < 1000), differences are negligible

#### Issue: `Error: FFTW not available`

**Cause:** fftw package not properly installed.

**Solution:**
```r
# Reinstall fftw
install.packages("fftw", type="source")

# Or use alternative CRAN mirror
install.packages("fftw", repos="https://cloud.r-project.org")
```

---

## Platform-Specific Notes

### Linux

**Python:**
- All features work out of the box
- Use system package manager for Python: `sudo apt-get install python3-pip`
- Virtual environments recommended: `python3 -m venv .venv`

**MATLAB:**
- No special considerations
- May need to set LD_LIBRARY_PATH if using custom MATLAB installation

**R:**
- Install from apt: `sudo apt-get install r-base r-base-dev`
- Build tools needed for some packages: `sudo apt-get install build-essential`

### macOS

**Python:**
- Use Homebrew Python for best results: `brew install python@3.11`
- On Apple Silicon (M1/M2/M3), ensure ARM64 native Python for best performance
- Check architecture: `python -c "import platform; print(platform.machine())"`
  - Should show `arm64` on Apple Silicon

**MATLAB:**
- MATLAB runs natively on Apple Silicon (M1+) as of R2023b
- Older versions use Rosetta 2 (slight performance penalty)

**R:**
- Install from CRAN or Homebrew: `brew install r`
- XCode Command Line Tools required: `xcode-select --install`

### Windows

**Python:**
- PowerShell recommended over CMD for better Unicode support
- Consider WSL (Windows Subsystem for Linux) for better performance:
  ```bash
  wsl --install
  wsl
  cd /mnt/c/path/to/fastLPR_py
  ```
- If using Anaconda, activate environment before installing:
  ```bash
  conda activate myenv
  pip install -e .
  ```

**MATLAB:**
- No special considerations
- Ensure MATLAB is in PATH for command-line usage

**R:**
- Install Rtools if building packages from source
- Download from: https://cran.r-project.org/bin/windows/Rtools/
- RStudio recommended for interactive use

---

## Performance Optimization

### Python

#### Option 1: Install Numba (Easy)

Numba provides JIT compilation for 10-20% speedup:

```bash
pip install numba
```

The code automatically uses Numba if available.

#### Option 2: Use OpenBLAS (AMD CPUs)

For AMD Ryzen/Threadripper CPUs, OpenBLAS is 2-5x faster than Intel MKL:

```bash
# Using conda (recommended)
conda install numpy "libblas=*=*openblas"

# Using pip
pip install numpy openblas

# Verify
python -c "import numpy as np; np.show_config()"
```

#### Option 3: GPU Acceleration (Experimental)

For very large datasets (N > 100,000), consider GPU:

```bash
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

Modify code to use CuPy arrays:
```python
import cupy as cp
x = cp.asarray(x)  # Transfer to GPU
```

### MATLAB

#### Vectorization

Ensure all loops are vectorized. Use:
```matlab
% Check for vectorization opportunities
profile on
regs = cv_fastLPR(x, y, hlist, opt);
profile viewer
```

Look for functions spending >10% time in loops.

#### Parallel Computing

If Parallel Computing Toolbox available:
```matlab
opt.use_parallel = true;
parpool(4);  % Use 4 workers
regs = cv_fastLPR(x, y, hlist, opt);
```

### R

#### Use Matrix Package

Always use sparse matrices for large data:
```r
library(Matrix)
X_sparse <- Matrix(X, sparse=TRUE)
```

#### Vectorize Operations

Avoid loops where possible:
```r
# Slow
for (i in 1:n) {
    result[i] <- compute(data[i])
}

# Fast
result <- sapply(data, compute)
```

---

## Docker Installation (Advanced)

For reproducible environments across platforms:

### Python Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY fastLPR_py/pyproject.toml /app/
RUN pip install --no-cache-dir numpy>=1.23 scipy>=1.10 matplotlib>=3.7

# Copy fastLPR code
COPY fastLPR_py /app/fastLPR_py

# Install fastLPR
RUN cd fastLPR_py && pip install -e .

# Set working directory to examples
WORKDIR /app/fastLPR_py/examples

CMD ["python", "example_fig2_fastkde.py"]
```

Build and run:
```bash
docker build -t fastlpr-python .
docker run -v $(pwd)/fig:/app/fig fastlpr-python
```

### MATLAB Dockerfile

Requires MATLAB license. Example using MATLAB Runtime:

```dockerfile
FROM mathworks/matlab-runtime:r2023b

WORKDIR /app
COPY fastLPR /app/fastLPR

# Add to MATLAB path
ENV MATLABPATH=/app/fastLPR/utility:/app/fastLPR/utility/core

CMD ["/app/fastLPR/example/reproduce_all_figures.m"]
```

---

## Uninstallation

### Python

```bash
# Uninstall package
pip uninstall fastlpr

# Remove virtual environment
deactivate
rm -rf .venv  # Linux/macOS
rmdir /s .venv  # Windows
```

### MATLAB

```matlab
% Remove from path
rmpath(genpath('path/to/fastLPR'));
savepath;  % Make permanent
```

### R

```r
# Uninstall packages
remove.packages(c("Matrix", "fftw", "interp", "akima"))

# Clean workspace
rm(list=ls())
```

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check Documentation:**
   - Python: `fastLPR_py/README.md`
   - MATLAB: `fastLPR/README.md`
   - R: `fastLPR_R/README.md`

2. **Check Module Documentation:**
   - Python: `fastLPR_py/CLAUDE.md`
   - MATLAB: `fastLPR/CLAUDE.md`
   - R: `fastLPR_R/CLAUDE.md`

3. **Review Examples:**
   - All examples have fixed random seeds and are fully reproducible
   - Compare your code with working examples

4. **Search Issues:** Check TODO.md for known issues

5. **Report Bug:**
   - Open GitHub issue
   - Include: OS, Python/MATLAB/R version, error message, minimal reproducible example

---

## Next Steps

After successful installation:

### For Python Users

1. Read the quick start guide: `fastLPR_py/README.md`
2. Run examples: `cd examples && python reproduce_all_figures.py`
3. Read API documentation: `help(cv_fastlpr)` in Python
4. Check verification: `fastLPR_py/tests/TEST_SUMMARY.md`

### For MATLAB Users

1. Read the user guide: `fastLPR/README.md`
2. Run examples: `cd example && reproduce_all_figures`
3. Read API documentation: `help cv_fastLPR` in MATLAB
4. Run test suite: `cd tests && run_all`

### For R Users

1. Read the README: `fastLPR_R/README.md`
2. Run basic examples: `cd examples && source("example_kde_1d.R")`
3. Check limitations: `fastLPR_R/CLAUDE.md`
4. Consider using Python via reticulate for full functionality

---

**Last Updated:** 2025-11-20
**Version:** 1.0.0
**Status:** Ready for JSS Submission

**Authors:**
- Ying Wang (yingwangrigel@gmail.com)
- Min Li (minli.231314@gmail.com)

**License:** GPL-3.0-or-later
