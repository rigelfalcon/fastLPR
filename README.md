# fastLPR: Fast Local Polynomial Regression via NUFFT

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Fast nonparametric regression and kernel density estimation using Non-Uniform Fast Fourier Transform (NUFFT) for MATLAB, Python, and R.

## Features

- **O(N + M log M) complexity** - Scales efficiently to large datasets
- **Multi-language support** - MATLAB (reference), Python, and R implementations
- **Flexible regression orders** - Local constant (NW), local linear, local quadratic
- **Automatic bandwidth selection** - GCV for regression, LCV for density estimation
- **Complex-valued data support** - Handles complex predictors and responses
- **Heteroscedastic variance estimation** - Estimates conditional variance

## Installation

### MATLAB

```matlab
cd fastLPR
fastLPR_setup
```

### Python

```bash
cd fastLPR_py
uv pip install -e .
```

### R

```r
# From source
setwd("fastLPR_R")
source("setup.R")

# After CRAN acceptance:
# install.packages("fastlpr")
```

## Quick Start

### MATLAB

```matlab
% 1D regression with automatic bandwidth selection
x = rand(500, 1);
y = sin(2*pi*x) + 0.1*randn(500, 1);
hlist = get_hList(20, [0.01, 1], @logspace);
opt.order = 1;  % Local linear
regs = cv_fastLPR(x, y, hlist, opt);
fastLPR_plot(regs.fpp_yhat);
```

### Python

```python
import numpy as np
from fastlpr import cv_fastlpr, get_hlist

x = np.random.rand(500, 1)
y = np.sin(2*np.pi*x) + 0.1*np.random.randn(500, 1)
hlist = get_hlist(20, [0.01, 1.0])
regs = cv_fastlpr(x, y, hlist, {'order': 1})
```

### R

```r
source("setup.R")
x <- matrix(runif(500), ncol=1)
y <- sin(2*pi*x) + 0.1*rnorm(500)
hlist <- get_hlist(20, c(0.01, 1.0))
regs <- cv_fastlpr(x, y, hlist, list(order=1))
```

## Documentation

- **MATLAB**: See `fastLPR/README.md` and function help texts
- **Python**: See `fastLPR_py/README.md` and docstrings
- **R**: See `fastLPR_R/README.md` and roxygen2 documentation

## Examples

Each implementation includes reproducible examples for the JSS paper figures:

```
fastLPR/example/reproduce_all_figures.m
fastLPR_py/examples/reproduce_all_figures.py
fastLPR_R/inst/examples/reproduce_all_figures.R
```

## Citation

If you use fastLPR in your research, please cite:

```bibtex
@article{wang2025fastlpr,
  title = {{fastLPR}: Fast Local Polynomial Regression via {NUFFT} in {MATLAB}, {Python}, and {R}},
  author = {Wang, Ying and Li, Min and Paz-Linares, Deirel and Valdes-Sosa, Pedro A.},
  journal = {Journal of Statistical Software},
  year = {2025},
  volume = {XXX},
  number = {XX},
  pages = {1--XX},
  doi = {10.18637/jss.vXXX.iXX}
}
```

## License

GPL-3.0-or-later. See [LICENSE](fastLPR/LICENSE) for details.

## Authors

- Ying Wang (yingwangrigel@gmail.com)
- Min Li (minli.231314@gmail.com)
- Deirel Paz-Linares
- Pedro A. Valdes-Sosa
