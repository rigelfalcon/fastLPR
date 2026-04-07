# fastLPR: Fast Local Polynomial Regression via NUFFT (MATLAB)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a+-orange.svg)](https://www.mathworks.com/products/matlab.html)

## Overview

**fastLPR** is a production-ready MATLAB toolbox for fast local polynomial regression and kernel density estimation on large scattered datasets. The package reduces computational complexity from O(N²) to O(N + M log M) using a custom Non-Uniform Fast Fourier Transform (NUFFT) with Gaussian gridding.

**Key Features:**
- Fast kernel regression and density estimation (10-100× speedup over existing packages)
- Supports scattered data in arbitrary dimensions (1D, 2D, 3D)
- Complex-valued data support (unique among nonparametric regression packages)
- Heteroscedastic variance estimation with log-space stability
- Automatic bandwidth selection via GCV/LCV with 1-SE rule
- Comprehensive test suite (12 tests, 100% pass rate)

## Installation

### Option 1: Quick Setup
```matlab
cd fastLPR
fastlpr_setup();
```

### Option 2: Toolbox Installer
```matlab
cd fastLPR
install_fastlpr();  % Prompts to save path
```

### Verify Installation
```matlab
help fastLPR
ver fastLPR
```

## Quick Start

### 1D Local Linear Regression
```matlab
% Generate test data
x = rand(500, 1);
y = sin(2*pi*x) + 0.1*randn(500, 1);

% Fast regression with automatic bandwidth selection
hlist = get_hList(20, [0.01, 1], @logspace);
opt.order = 1;  % Local linear
regs = cv_fastLPR(x, y, hlist, opt);

% Visualize
fastLPR_plot(regs.fpp_yhat);
```

### 1D Kernel Density Estimation
```matlab
% Generate bimodal data
x = [randn(100,1)-2; randn(100,1)+2];

% Fast KDE with automatic bandwidth selection
hlist = get_hList(20, [0.01, 1], @logspace);
kde = cv_fastKDE(x, hlist);

% Visualize
fastKDE_plot(kde);
```

## Examples

Reproduce all JSS paper figures:
```matlab
cd fastLPR/example
reproduce_all_figures  % Generates the paper figures
```

Individual examples:
- `example_fig2_fastkde.m` - 1D kernel density estimation
- `example_fig3_boundary_comparison.m` - Boundary effect comparison (NW vs LL vs LQ)
- `example_fig4_complex.m` - Complex-valued regression
- `example_fig5_heteroscedasticity.m` - Heteroscedastic regression with intervals (CI and PI)
- `example_fig6_applications.m` - Real-world qEEG application

## Testing

Run all tests:
```matlab
cd tests
run_all  % 12 tests, should all pass
```

Key test:
```matlab
test_fastlpr_vs_naive_nw  % Validates accuracy vs naive Nadaraya-Watson
```

## Documentation

- **Getting Started:** `help fastLPR`
- **API Reference:** `help function_name` (e.g., `help cv_fastLPR`)
- **Examples:** See `example/` directory
- **Paper:** Wang et al. (2025), Journal of Statistical Software

## Requirements

- MATLAB R2020a or later
- No additional toolboxes required (base MATLAB only)

## Citation

If you use fastLPR in your research, please cite:

```bibtex
@article{wang2025fastlpr,
  title={fastLPR: Fast Local Polynomial Regression via NUFFT in MATLAB, Python, and R},
  author={Wang, Ying and Li, Min and Paz-Linares, Deirel and Valdes-Sosa, Pedro A.},
  journal={Journal of Statistical Software},
  year={2025},
  note={In preparation}
}
```

## License

GPL-3.0 License. See [LICENSE](LICENSE) for details.

## Authors

- Ying Wang (yingwangrigel@gmail.com)
- Min Li (minli.231314@gmail.com)
- Deirel Paz-Linares
- Pedro A. Valdes-Sosa

## Links

- **GitHub (source):** https://github.com/rigelfalcon/fastLPR
- **PyPI (Python):** release forthcoming
- **CRAN (R):** release forthcoming
- **Paper:** Journal of Statistical Software (under review)
