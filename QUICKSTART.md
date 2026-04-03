# Quick Start Guide - fastLPR

**5-minute setup guide for each implementation**

---

## Python (Recommended)

### 1. Install

```bash
cd fastLPR_py
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Test

```python
import numpy as np
from fastlpr import cv_fastlpr, get_hlist

x = np.random.rand(100, 1)
y = x**2 + 0.1*np.random.randn(100, 1)
hlist = get_hlist(10, [0.01, 1.0])
regs = cv_fastlpr(x, y, hlist, {'order': 1})
print(f"Selected bandwidth: {regs.bandwidth}")
```

### 3. Run Examples

```bash
cd examples
python example_fig2_fastkde.py
```

**Full Guide:** See [INSTALL.md](INSTALL.md#python-installation)

---

## MATLAB

### 1. Setup

```matlab
cd fastLPR
fastLPR_setup();
```

### 2. Test

```matlab
x = rand(500, 1);
y = sin(2*pi*x) + 0.1*randn(500, 1);
hlist = get_hList(20, [0.01, 1], @logspace);
opt.order = 1;
regs = cv_fastLPR(x, y, hlist, opt);
fastLPR_plot(regs.fpp_yhat);
```

### 3. Run Examples

```matlab
cd example
example_fig2_fastkde
```

**Full Guide:** See [INSTALL.md](INSTALL.md#matlab-installation)

---

## R (Limited: 1D/2D only)

### 1. Setup

```r
setwd("fastLPR_R")
install.packages(c("Matrix", "fftw", "interp", "akima"))
source("setup.R")
```

### 2. Test

```r
x <- matrix(rnorm(200), ncol=1)
hlist <- get_hlist(10, c(0.1, 1.0))
kde <- cv_fastkde(x, hlist)
print(paste("Selected bandwidth:", kde$h_opt))
```

### 3. Run Examples

```r
setwd("examples")
source("example_kde_1d.R")
```

**Note:** For full functionality, use Python via `reticulate`:
```r
library(reticulate)
fastlpr <- import("fastlpr")
```

**Full Guide:** See [INSTALL.md](INSTALL.md#r-installation)

---

## Common Issues

| Issue | Solution |
|-------|----------|
| **Python:** `ModuleNotFoundError: No module named 'fastlpr'` | Run `pip install -e .` from `fastLPR_py/` |
| **Python:** Slow on AMD CPU | Install OpenBLAS: `conda install numpy "libblas=*=*openblas"` |
| **MATLAB:** `Undefined function 'cv_fastLPR'` | Run `fastLPR_setup()` |
| **MATLAB:** Very slow | Check vectorization with `profile viewer` |
| **R:** Missing packages | Install: `install.packages(c("Matrix", "fftw", "interp", "akima"))` |
| **R:** Limited functionality | Use Python via reticulate: `library(reticulate); fastlpr <- import("fastlpr")` |

**Full Troubleshooting:** See [INSTALL.md](INSTALL.md#troubleshooting)

---

## Implementation Comparison

| Feature | Python | MATLAB | R |
|---------|--------|--------|---|
| **Status** | ✅ Production | ✅ Production | ⚠️ 95% Complete |
| **Dimensions** | 1D/2D/3D | 1D/2D/3D | 1D/2D only |
| **Orders** | 0/1/2 | 0/1/2 | 0/1/2 |
| **Complex Data** | ✅ | ✅ | ✅ |
| **GCV/LCV** | ✅ | ✅ | ✅ |
| **Variance Estimation** | ✅ | ✅ | ⚠️ 1D only |
| **Performance** | Fast | Fastest | Fast |
| **Dependencies** | NumPy, SciPy | None | Matrix, fftw |
| **Tests** | ✅ Full | ✅ Full | ⚠️ Manual |

---

## Where to Get Help

1. **Installation Issues:** [INSTALL.md](INSTALL.md)
2. **Usage Examples:** See `examples/` directory in each implementation
3. **API Reference:**
   - Python: `help(cv_fastlpr)` or see `fastLPR_py/CLAUDE.md`
   - MATLAB: `help cv_fastLPR` or see `fastLPR/CLAUDE.md`
   - R: See `fastLPR_R/CLAUDE.md`
4. **Known Issues:** See `TODO.md`

---

**Last Updated:** 2025-11-20
**Authors:** Ying Wang, Min Li
**License:** GPL-3.0-or-later
