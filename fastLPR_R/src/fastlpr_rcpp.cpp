// fastlpr_rcpp.cpp - High-performance C++ implementations for fastLPR
//
// This file provides optimized implementations of key numerical operations:
// 1. N-dimensional FFT (forward and inverse) with OpenMP parallelization
// 2. N-dimensional linear interpolation with OpenMP parallelization
// 3. Fast array operations (permutation, broadcasting)
// 4. Batch convolution for multiple bandwidths
//
// Design principles:
// - Support arbitrary dimensions (1D, 2D, 3D, etc.)
// - OpenMP parallelization for multi-core performance (like MATLAB)
// - Clean error handling with informative messages
// - Match R's array layout (column-major order)
// - Provide fallback to pure R implementations
//
// OpenMP support: Enabled via Rcpp::plugins(openmp)
// To control threads: armadillo_set_number_of_omp_threads(n)

#include <RcppArmadillo.h>
#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

#ifdef _OPENMP
#include <omp.h>
#endif

// Direct FFTW3 for single-precision (float32) FFT path
// When ARMA_USE_FFTW3 is defined, fftw3.h is available via Armadillo's dependency.
// We use fftwf_* (single-precision) functions directly for accuracy <= 4.
#ifdef ARMA_USE_FFTW3
#include <fftw3.h>
#endif

using namespace Rcpp;
using namespace arma;


// =============================================================================
// N-DIMENSIONAL FFT
// =============================================================================
//
// PERFORMANCE NOTE:
// Standalone FFT wrappers below are slower than R's base fft()/mvfft() when
// called from R due to R<->C++ data copy overhead. These are kept for
// internal C++ use (inside rcpp_nufft_type1, rcpp_conv_nd_full) where
// there is no R<->C++ crossing.
// =============================================================================

// Apply 1D FFT along a specific axis of an N-dimensional array
// NOTE: Slower than R fft() when called from R. Use base R fft() instead.
// [[Rcpp::export]]
ComplexVector rcpp_fft_axis(ComplexVector arr, IntegerVector dims, int axis, bool inverse = false) {
    int ndim = dims.size();
    if (axis < 0 || axis >= ndim) {
        stop("axis must be in range [0, ndim-1]");
    }
    
    size_t total = 1;
    std::vector<size_t> d(ndim);
    for (int i = 0; i < ndim; ++i) {
        d[i] = static_cast<size_t>(dims[i]);
        total *= d[i];
    }
    
    if ((size_t)arr.size() != total) {
        stop("Array size does not match dimensions");
    }
    
    ComplexVector result(total);
    size_t axis_len = d[axis];
    
    // Compute strides (column-major)
    std::vector<size_t> strides(ndim);
    strides[0] = 1;
    for (int i = 1; i < ndim; ++i) {
        strides[i] = strides[i-1] * d[i-1];
    }
    
    size_t num_ffts = total / axis_len;
    
    for (size_t fft_idx = 0; fft_idx < num_ffts; ++fft_idx) {
        // Compute starting position for this FFT
        std::vector<size_t> base_sub(ndim, 0);
        size_t temp = fft_idx;
        for (int i = ndim - 1; i >= 0; --i) {
            if (i != axis) {
                base_sub[i] = temp % d[i];
                temp /= d[i];
            }
        }
        
        // Extract 1D slice
        cx_vec slice(axis_len);
        for (size_t k = 0; k < axis_len; ++k) {
            size_t idx = 0;
            for (int i = 0; i < ndim; ++i) {
                if (i == axis) {
                    idx += k * strides[i];
                } else {
                    idx += base_sub[i] * strides[i];
                }
            }
            slice(k) = std::complex<double>(arr[idx].r, arr[idx].i);
        }
        
        // Apply FFT
        cx_vec fft_result;
        if (inverse) {
            fft_result = arma::ifft(slice);
        } else {
            fft_result = arma::fft(slice);
        }
        
        // Store result
        for (size_t k = 0; k < axis_len; ++k) {
            size_t idx = 0;
            for (int i = 0; i < ndim; ++i) {
                if (i == axis) {
                    idx += k * strides[i];
                } else {
                    idx += base_sub[i] * strides[i];
                }
            }
            result[idx] = Rcomplex{{fft_result(k).real(), fft_result(k).imag()}};
        }
    }
    
    result.attr("dim") = dims;
    return result;
}

// N-dimensional FFT
// [[Rcpp::export]]
ComplexVector rcpp_fft_nd(ComplexVector arr, IntegerVector dims, bool inverse = false) {
    int ndim = dims.size();
    ComplexVector result = clone(arr);
    result.attr("dim") = dims;
    
    for (int axis = ndim - 1; axis >= 0; --axis) {
        result = rcpp_fft_axis(result, dims, axis, inverse);
    }
    
    return result;
}

// Real input wrapper
// [[Rcpp::export]]
ComplexVector rcpp_fft_nd_real(NumericVector arr, IntegerVector dims, bool inverse = false) {
    ComplexVector carr(arr.size());
    for (int i = 0; i < arr.size(); ++i) {
        carr[i] = Rcomplex{{arr[i], 0.0}};
    }
    carr.attr("dim") = dims;
    return rcpp_fft_nd(carr, dims, inverse);
}


// =============================================================================
// N-DIMENSIONAL LINEAR INTERPOLATION
// =============================================================================

// [[Rcpp::export]]
NumericVector rcpp_interp_nd(List grid_vals, NumericVector arr, NumericMatrix points) {
    int ndim = grid_vals.size();
    int n_points = points.nrow();
    
    if (points.ncol() != ndim) {
        stop("points must have same number of columns as dimensions");
    }
    
    std::vector<std::vector<double>> grids(ndim);
    std::vector<size_t> dims(ndim);
    size_t total = 1;
    
    for (int d = 0; d < ndim; ++d) {
        NumericVector gv = grid_vals[d];
        grids[d] = std::vector<double>(gv.begin(), gv.end());
        dims[d] = grids[d].size();
        total *= dims[d];
    }
    
    if ((size_t)arr.size() != total) {
        stop("Array size does not match grid dimensions");
    }
    
    std::vector<size_t> strides(ndim);
    strides[0] = 1;
    for (int d = 1; d < ndim; ++d) {
        strides[d] = strides[d-1] * dims[d-1];
    }
    
    NumericVector result(n_points);
    
    for (int p = 0; p < n_points; ++p) {
        std::vector<int> lo_idx(ndim);
        std::vector<double> t(ndim);
        
        for (int d = 0; d < ndim; ++d) {
            double x = points(p, d);
            const std::vector<double>& g = grids[d];
            
            auto it = std::lower_bound(g.begin(), g.end(), x);
            int idx = static_cast<int>(it - g.begin());
            
            if (idx <= 0) {
                lo_idx[d] = 0;
                t[d] = 0.0;
            } else if (idx >= (int)g.size()) {
                lo_idx[d] = g.size() - 2;
                t[d] = 1.0;
            } else {
                lo_idx[d] = idx - 1;
                double x0 = g[lo_idx[d]];
                double x1 = g[lo_idx[d] + 1];
                t[d] = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
            }
        }
        
        int n_corners = 1 << ndim;
        double value = 0.0;
        
        for (int corner = 0; corner < n_corners; ++corner) {
            double weight = 1.0;
            size_t idx = 0;
            
            for (int d = 0; d < ndim; ++d) {
                int bit = (corner >> d) & 1;
                int grid_idx = lo_idx[d] + bit;
                if (grid_idx >= (int)dims[d]) grid_idx = dims[d] - 1;
                
                idx += grid_idx * strides[d];
                weight *= bit ? t[d] : (1.0 - t[d]);
            }
            
            value += weight * arr[idx];
        }
        
        result[p] = value;
    }
    
    return result;
}

// Complex version - operates directly on Rcomplex (no real/imag split)
// [[Rcpp::export]]
ComplexVector rcpp_interp_nd_complex(List grid_vals, ComplexVector arr, NumericMatrix points) {
    int ndim = grid_vals.size();
    int n_points = points.nrow();

    std::vector<std::vector<double>> grids(ndim);
    std::vector<size_t> dims(ndim);
    size_t total = 1;

    for (int d = 0; d < ndim; ++d) {
        NumericVector gv = grid_vals[d];
        grids[d] = std::vector<double>(gv.begin(), gv.end());
        dims[d] = grids[d].size();
        total *= dims[d];
    }

    if ((size_t)arr.size() != total) {
        stop("Array size does not match grid dimensions");
    }

    std::vector<size_t> strides(ndim);
    strides[0] = 1;
    for (int d = 1; d < ndim; ++d) {
        strides[d] = strides[d-1] * dims[d-1];
    }

    ComplexVector result(n_points);

    for (int p = 0; p < n_points; ++p) {
        std::vector<int> lo_idx(ndim);
        std::vector<double> t(ndim);

        for (int d = 0; d < ndim; ++d) {
            double x = points(p, d);
            const std::vector<double>& g = grids[d];

            auto it = std::lower_bound(g.begin(), g.end(), x);
            int idx = static_cast<int>(it - g.begin());

            if (idx <= 0) {
                lo_idx[d] = 0;
                t[d] = 0.0;
            } else if (idx >= (int)g.size()) {
                lo_idx[d] = g.size() - 2;
                t[d] = 1.0;
            } else {
                lo_idx[d] = idx - 1;
                double x0 = g[lo_idx[d]];
                double x1 = g[lo_idx[d] + 1];
                t[d] = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
            }
        }

        int n_corners = 1 << ndim;
        double value_r = 0.0, value_i = 0.0;

        for (int corner = 0; corner < n_corners; ++corner) {
            double weight = 1.0;
            size_t idx = 0;

            for (int d = 0; d < ndim; ++d) {
                int bit = (corner >> d) & 1;
                int grid_idx = lo_idx[d] + bit;
                if (grid_idx >= (int)dims[d]) grid_idx = dims[d] - 1;

                idx += grid_idx * strides[d];
                weight *= bit ? t[d] : (1.0 - t[d]);
            }

            // Direct complex accumulation (no intermediate arrays)
            value_r += weight * arr[idx].r;
            value_i += weight * arr[idx].i;
        }

        result[p] = Rcomplex{{value_r, value_i}};
    }

    return result;
}


// =============================================================================
// FAST ARRAY OPERATIONS
// =============================================================================

// [[Rcpp::export]]
NumericVector rcpp_aperm(NumericVector arr, IntegerVector dims, IntegerVector perm) {
    int ndim = dims.size();
    if (perm.size() != ndim) stop("perm must have same length as dims");
    
    std::vector<int> p(ndim);
    for (int i = 0; i < ndim; ++i) {
        p[i] = perm[i] - 1;  // R is 1-indexed
        if (p[i] < 0 || p[i] >= ndim) stop("Invalid permutation index");
    }
    
    std::vector<size_t> in_dims(ndim), out_dims(ndim);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        in_dims[i] = dims[i];
        total *= dims[i];
        out_dims[i] = in_dims[p[i]];
    }
    
    std::vector<size_t> in_strides(ndim), out_strides(ndim);
    in_strides[0] = out_strides[0] = 1;
    for (int i = 1; i < ndim; ++i) {
        in_strides[i] = in_strides[i-1] * in_dims[i-1];
        out_strides[i] = out_strides[i-1] * out_dims[i-1];
    }
    
    NumericVector result(total);
    
    for (size_t out_idx = 0; out_idx < total; ++out_idx) {
        std::vector<size_t> out_sub(ndim), in_sub(ndim);
        size_t temp = out_idx;
        for (int d = 0; d < ndim; ++d) {
            out_sub[d] = temp % out_dims[d];
            temp /= out_dims[d];
        }
        
        for (int d = 0; d < ndim; ++d) {
            in_sub[p[d]] = out_sub[d];
        }
        
        size_t in_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            in_idx += in_sub[d] * in_strides[d];
        }
        
        result[out_idx] = arr[in_idx];
    }
    
    IntegerVector out_dims_r(ndim);
    for (int i = 0; i < ndim; ++i) out_dims_r[i] = out_dims[i];
    result.attr("dim") = out_dims_r;
    
    return result;
}

// Complex version - operates directly on Rcomplex (no real/imag split)
// [[Rcpp::export]]
ComplexVector rcpp_aperm_complex(ComplexVector arr, IntegerVector dims, IntegerVector perm) {
    int ndim = dims.size();
    if (perm.size() != ndim) stop("perm must have same length as dims");

    std::vector<int> p(ndim);
    for (int i = 0; i < ndim; ++i) {
        p[i] = perm[i] - 1;  // R is 1-indexed
        if (p[i] < 0 || p[i] >= ndim) stop("Invalid permutation index");
    }

    std::vector<size_t> in_dims(ndim), out_dims(ndim);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        in_dims[i] = dims[i];
        total *= dims[i];
        out_dims[i] = in_dims[p[i]];
    }

    std::vector<size_t> in_strides(ndim), out_strides(ndim);
    in_strides[0] = out_strides[0] = 1;
    for (int i = 1; i < ndim; ++i) {
        in_strides[i] = in_strides[i-1] * in_dims[i-1];
        out_strides[i] = out_strides[i-1] * out_dims[i-1];
    }

    ComplexVector result(total);

    for (size_t out_idx = 0; out_idx < total; ++out_idx) {
        std::vector<size_t> out_sub(ndim), in_sub(ndim);
        size_t temp = out_idx;
        for (int d = 0; d < ndim; ++d) {
            out_sub[d] = temp % out_dims[d];
            temp /= out_dims[d];
        }

        for (int d = 0; d < ndim; ++d) {
            in_sub[p[d]] = out_sub[d];
        }

        size_t in_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            in_idx += in_sub[d] * in_strides[d];
        }

        // Direct complex copy (no intermediate arrays)
        result[out_idx] = arr[in_idx];
    }

    IntegerVector out_dims_r(ndim);
    for (int i = 0; i < ndim; ++i) out_dims_r[i] = out_dims[i];
    result.attr("dim") = out_dims_r;

    return result;
}


// =============================================================================
// OPENMP THREAD CONTROL
// =============================================================================

// [[Rcpp::export]]
int rcpp_get_num_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// [[Rcpp::export]]
void rcpp_set_num_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
}


// =============================================================================
// PARALLEL BATCH INTERPOLATION (CRITICAL FOR DOF COMPUTATION)
// =============================================================================
// UNIFIED N-DIMENSIONAL BATCH INTERPOLATION (LIKE MATLAB griddedInterpolant)
// =============================================================================
//
// This is the unified version that handles ALL dimensions (1D, 2D, 3D, etc.)
// using loops, exactly like MATLAB's griddedInterpolant works internally.
//
// Key optimizations:
//   1. findInterval computed ONCE for each dimension (O(n_query * log(n_grid)))
//   2. OpenMP parallelization over bandwidths
//   3. Loop over 2^dx corners instead of dimension-specific code
//
// Legacy rcpp_interp_batch_1d and rcpp_interp_batch_2d functions archived at:
// dev/archive/fastLPR_R_interp_20260109/rcpp_interp_batch_legacy.cpp
// =============================================================================

// Unified N-dimensional batch linear interpolation
// grid_vectors: List of 1D grid vectors (one per dimension)
// values: Flattened array (N1 * N2 * ... * Nd * n_bandwidth)
// dims: [N1, N2, ..., Nd, n_bandwidth]
// query_points: (n_query, dx) matrix
//
// Returns: (n_query, n_bandwidth) matrix of interpolated values
//
// [[Rcpp::export]]
NumericMatrix rcpp_interp_batch_nd(List grid_vectors, NumericVector values,
                                    IntegerVector dims, NumericMatrix query_points) {
    int dx = grid_vectors.size();  // Number of spatial dimensions
    int n_query = query_points.nrow();

    if (query_points.ncol() != dx) {
        stop("query_points must have %d columns (got %d)", dx, (int)query_points.ncol());
    }

    // Parse dimensions: [N1, N2, ..., Nd, n_bandwidth]
    if (dims.size() != dx + 1) {
        stop("dims must have length dx + 1 (got %d, expected %d)", (int)dims.size(), dx + 1);
    }

    std::vector<int> N(dx);
    size_t N_total = 1;
    for (int d = 0; d < dx; ++d) {
        N[d] = dims[d];
        N_total *= N[d];
    }
    int n_bandwidth = dims[dx];

    // Validate values size
    size_t expected_size = N_total * n_bandwidth;
    if ((size_t)values.size() != expected_size) {
        stop("values size mismatch: expected %d, got %d", (int)expected_size, (int)values.size());
    }

    // Extract grid vectors into C++ vectors
    std::vector<std::vector<double>> grids(dx);
    for (int d = 0; d < dx; ++d) {
        NumericVector gv = grid_vectors[d];
        if ((int)gv.size() != N[d]) {
            stop("grid_vectors[%d] size mismatch: expected %d, got %d", d, N[d], (int)gv.size());
        }
        grids[d] = std::vector<double>(gv.begin(), gv.end());
    }

    // Compute strides (column-major order)
    std::vector<size_t> strides(dx);
    strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        strides[d] = strides[d-1] * N[d-1];
    }

    // Step 1: Compute findInterval ONCE for all query points (O(n_query * log(n)))
    // For each dimension d, store: lo_idx[q][d], t[q][d]
    std::vector<std::vector<int>> lo_idx(n_query, std::vector<int>(dx));
    std::vector<std::vector<double>> t(n_query, std::vector<double>(dx));

    for (int d = 0; d < dx; ++d) {
        const std::vector<double>& g = grids[d];
        int ng = N[d];

        for (int q = 0; q < n_query; ++q) {
            double x = query_points(q, d);

            // Binary search for interval
            int lo = 0, hi = ng - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (g[mid] > x) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }

            // Handle boundary: clamp to valid range
            if (lo >= ng - 1) lo = ng - 2;
            if (lo < 0) lo = 0;
            lo_idx[q][d] = lo;

            // Compute interpolation weight (can be < 0 or > 1 for extrapolation)
            double x0 = g[lo];
            double x1 = g[lo + 1];
            t[q][d] = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
        }
    }

    NumericMatrix result(n_query, n_bandwidth);

    // Step 2: Interpolate all bandwidths using precomputed indices (parallel)
    // Loop over 2^dx corners for linear interpolation in any dimension
    int n_corners = 1 << dx;  // 2^dx corners

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int h = 0; h < n_bandwidth; ++h) {
        // Offset into values array for this bandwidth
        size_t h_offset = (size_t)h * N_total;

        for (int q = 0; q < n_query; ++q) {
            double val = 0.0;

            // Loop over all 2^dx corners of the interpolation cell
            for (int corner = 0; corner < n_corners; ++corner) {
                double weight = 1.0;
                size_t idx = 0;

                // For each dimension, choose lo or hi based on corner bit
                for (int d = 0; d < dx; ++d) {
                    int bit = (corner >> d) & 1;  // 0 = lo, 1 = hi
                    int grid_idx = lo_idx[q][d] + bit;

                    // Clamp to valid range
                    if (grid_idx >= N[d]) grid_idx = N[d] - 1;
                    if (grid_idx < 0) grid_idx = 0;

                    idx += grid_idx * strides[d];
                    weight *= bit ? t[q][d] : (1.0 - t[q][d]);
                }

                val += weight * values[h_offset + idx];
            }

            result(q, h) = val;
        }
    }

    return result;
}


// =============================================================================
// PARALLEL FFT OPERATIONS
// =============================================================================
//
// PERFORMANCE NOTE:
// Standalone FFT wrappers are slower than R's base fft() when called from R
// due to R<->C++ data copy overhead. They ARE useful for internal C++ use
// (inside rcpp_nufft_type1, rcpp_conv_nd_full) where there's no R<->C++ crossing.
//
// DO NOT call rcpp_fft2d_batch/rcpp_fft3d_batch from R's design_matrix.R.
// Use base R fft() instead for standalone FFT operations.
// =============================================================================

// Parallel 3D FFT - process all bandwidths in parallel
// For 4D array (N1, N2, N3, dh), apply 3D FFT to each bandwidth slice
// This eliminates the expensive aperm() calls in R
// NOTE: Slower than R fft() when called from R. See note above.
// [[Rcpp::export]]
ComplexVector rcpp_fft3d_batch(ComplexVector arr, int n1, int n2, int n3, int dh,
                                bool inverse = false) {
    size_t total = (size_t)n1 * n2 * n3 * dh;
    if ((size_t)arr.size() != total) {
        stop("Array size mismatch: expected %d, got %d", (int)total, (int)arr.size());
    }
    
    ComplexVector result(total);
    size_t slice_size = (size_t)n1 * n2 * n3;
    
    // Process each bandwidth slice in parallel
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int h = 0; h < dh; ++h) {
        size_t offset = h * slice_size;
        
        // Working buffer for 3D FFT
        std::vector<std::complex<double>> buf(slice_size);
        
        // Copy slice to buffer
        for (size_t i = 0; i < slice_size; ++i) {
            buf[i] = std::complex<double>(arr[offset + i].r, arr[offset + i].i);
        }
        
        // Apply FFT along each dimension (like MATLAB: for ix=3:-1:1; m=fft(m,[],ix); end)
        
        // Axis 0 (n1): contiguous memory, easy
        cx_vec tmp1d(n1);
        for (int k = 0; k < n3; ++k) {
            for (int j = 0; j < n2; ++j) {
                // Extract column
                for (int i = 0; i < n1; ++i) {
                    tmp1d(i) = buf[i + j * n1 + k * n1 * n2];
                }
                // FFT (avoid ternary due to type mismatch)
                cx_vec fft_result;
                if (inverse) {
                    fft_result = arma::ifft(tmp1d);
                } else {
                    fft_result = arma::fft(tmp1d);
                }
                // Store
                for (int i = 0; i < n1; ++i) {
                    buf[i + j * n1 + k * n1 * n2] = fft_result(i);
                }
            }
        }
        
        // Axis 1 (n2): stride = n1
        cx_vec tmp2d(n2);
        for (int k = 0; k < n3; ++k) {
            for (int i = 0; i < n1; ++i) {
                for (int j = 0; j < n2; ++j) {
                    tmp2d(j) = buf[i + j * n1 + k * n1 * n2];
                }
                cx_vec fft_result;
                if (inverse) {
                    fft_result = arma::ifft(tmp2d);
                } else {
                    fft_result = arma::fft(tmp2d);
                }
                for (int j = 0; j < n2; ++j) {
                    buf[i + j * n1 + k * n1 * n2] = fft_result(j);
                }
            }
        }
        
        // Axis 2 (n3): stride = n1*n2
        cx_vec tmp3d(n3);
        for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
                for (int k = 0; k < n3; ++k) {
                    tmp3d(k) = buf[i + j * n1 + k * n1 * n2];
                }
                cx_vec fft_result;
                if (inverse) {
                    fft_result = arma::ifft(tmp3d);
                } else {
                    fft_result = arma::fft(tmp3d);
                }
                for (int k = 0; k < n3; ++k) {
                    buf[i + j * n1 + k * n1 * n2] = fft_result(k);
                }
            }
        }
        
        // Copy result back
        for (size_t i = 0; i < slice_size; ++i) {
            result[offset + i] = Rcomplex{{buf[i].real(), buf[i].imag()}};
        }
    }
    
    IntegerVector dims = IntegerVector::create(n1, n2, n3, dh);
    result.attr("dim") = dims;
    return result;
}

// Parallel 2D FFT - process slices in parallel
// For 3D array (N1, N2, n_slices), apply 2D FFT to each slice
// NOTE: Slower than R fft() when called from R. See PERFORMANCE NOTE above.
// [[Rcpp::export]]
ComplexVector rcpp_fft2d_batch(ComplexVector arr, int n1, int n2, int n_slices,
                                bool inverse = false) {
    size_t total = (size_t)n1 * n2 * n_slices;
    if ((size_t)arr.size() != total) {
        stop("Array size does not match dimensions");
    }
    
    ComplexVector result(total);
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s = 0; s < n_slices; ++s) {
        // Extract slice
        cx_mat slice(n1, n2);
        size_t offset = s * n1 * n2;
        for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
                size_t idx = offset + i + j * n1;
                slice(i, j) = std::complex<double>(arr[idx].r, arr[idx].i);
            }
        }
        
        // Apply 2D FFT
        cx_mat fft_result;
        if (inverse) {
            fft_result = arma::ifft2(slice);
        } else {
            fft_result = arma::fft2(slice);
        }
        
        // Store result
        for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
                size_t idx = offset + i + j * n1;
                result[idx] = Rcomplex{{fft_result(i, j).real(), fft_result(i, j).imag()}};
            }
        }
    }
    
    IntegerVector dims = IntegerVector::create(n1, n2, n_slices);
    result.attr("dim") = dims;
    return result;
}


// =============================================================================
// FAST CONVOLUTION BROADCAST (KEY FOR PERFORMANCE)
// =============================================================================

// Broadcast multiply: kdf (L, dh) * y_ft (L, dy) -> result (L, dh, dy)
// This replaces the expensive array(rep()) in R
// [[Rcpp::export]]
ComplexVector rcpp_broadcast_multiply(ComplexVector kdf, ComplexVector y_ft,
                                       int L, int dh, int dy) {
    if (kdf.size() != L * dh) {
        stop("kdf size mismatch: expected %d, got %d", (int)(L * dh), (int)kdf.size());
    }
    if (y_ft.size() != L * dy) {
        stop("y_ft size mismatch: expected %d, got %d", (int)(L * dy), (int)y_ft.size());
    }
    
    ComplexVector result(L * dh * dy);
    
    // Parallel over output: result[l, h, d] = kdf[l, h] * y_ft[l, d]
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int d = 0; d < dy; ++d) {
        for (int h = 0; h < dh; ++h) {
            for (int l = 0; l < L; ++l) {
                // Column-major indexing
                size_t kdf_idx = l + h * L;        // kdf[l, h]
                size_t y_idx = l + d * L;          // y_ft[l, d]
                size_t out_idx = l + h * L + d * L * dh;  // result[l, h, d]
                
                std::complex<double> k(kdf[kdf_idx].r, kdf[kdf_idx].i);
                std::complex<double> y(y_ft[y_idx].r, y_ft[y_idx].i);
                std::complex<double> r = k * y;
                
                result[out_idx] = Rcomplex{{r.real(), r.imag()}};
            }
        }
    }
    
    IntegerVector dims = IntegerVector::create(L, dh, dy);
    result.attr("dim") = dims;
    return result;
}

// Multi-dimensional broadcast multiply for 2D/3D spatial grids
// kdf (L1*L2, dh) * y_ft (L1*L2, dy) -> result (L1*L2, dh, dy)
// [[Rcpp::export]]
ComplexVector rcpp_broadcast_multiply_nd(ComplexVector kdf, ComplexVector y_ft,
                                          IntegerVector spatial_dims, int dh, int dy) {
    int L_spatial = 1;
    for (int i = 0; i < spatial_dims.size(); ++i) {
        L_spatial *= spatial_dims[i];
    }
    
    return rcpp_broadcast_multiply(kdf, y_ft, L_spatial, dh, dy);
}


// =============================================================================
// PARALLEL ARRAY OPERATIONS
// =============================================================================

// Parallel array extraction (subsetting)
// Extract subarray arr[start1:end1, start2:end2, :, :] 
// [[Rcpp::export]]
NumericVector rcpp_extract_subarray(NumericVector arr, IntegerVector dims,
                                     IntegerVector starts, IntegerVector ends) {
    int ndim = dims.size();
    if (starts.size() != ndim || ends.size() != ndim) {
        stop("starts and ends must have same length as dims");
    }
    
    // Compute output dimensions
    std::vector<int> out_dims(ndim);
    int out_total = 1;
    for (int d = 0; d < ndim; ++d) {
        out_dims[d] = ends[d] - starts[d] + 1;
        out_total *= out_dims[d];
    }
    
    // Compute strides
    std::vector<int> in_strides(ndim), out_strides(ndim);
    in_strides[0] = out_strides[0] = 1;
    for (int d = 1; d < ndim; ++d) {
        in_strides[d] = in_strides[d-1] * dims[d-1];
        out_strides[d] = out_strides[d-1] * out_dims[d-1];
    }
    
    NumericVector result(out_total);
    
    // Parallel extraction
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int out_idx = 0; out_idx < out_total; ++out_idx) {
        // Convert linear index to subscripts
        int temp = out_idx;
        int in_idx = 0;
        for (int d = 0; d < ndim; ++d) {
            int out_sub = temp % out_dims[d];
            temp /= out_dims[d];
            int in_sub = out_sub + starts[d] - 1;  // R is 1-indexed
            in_idx += in_sub * in_strides[d];
        }
        result[out_idx] = arr[in_idx];
    }
    
    IntegerVector out_dims_r(ndim);
    for (int d = 0; d < ndim; ++d) out_dims_r[d] = out_dims[d];
    result.attr("dim") = out_dims_r;
    
    return result;
}


// =============================================================================
// SINGLE-PRECISION FFT HELPERS (via FFTW3 fftwf_*)
// =============================================================================
//
// When accuracy <= 4, we can use float32/complex64 for FFT operations.
// This halves memory bandwidth and can provide ~1.5-2x speedup on FFT-heavy
// workloads. The approach:
//   1. Downcast complex<double> workspace to complex<float>
//   2. Run FFT/IFFT via fftwf_plan_dft_1d (single-precision FFTW3)
//   3. Upcast result back to complex<double>
//
// Only available when FFTW3 is linked (ARMA_USE_FFTW3 defined).
// Falls back to double-precision Armadillo FFT otherwise.

#ifdef ARMA_USE_FFTW3

// Thread-safety: FFTW3 plan creation/destruction is NOT thread-safe.
// We protect with a mutex when OpenMP is enabled.
#ifdef _OPENMP
#include <mutex>
static std::mutex fftw_plan_mutex;
#define FFTW_PLAN_LOCK() std::lock_guard<std::mutex> _fftw_lock(fftw_plan_mutex)
#else
#define FFTW_PLAN_LOCK() ((void)0)
#endif

// In-place 1D IFFT on a contiguous array of complex<float>, length n.
// Uses FFTW3 single-precision. Normalizes by 1/n (to match arma::ifft).
// NOTE: Use ::fftwf_plan_s* to disambiguate from arma::fftwf_plan (void* typedef).
static void fftwf_ifft_1d_inplace(std::complex<float>* data, int n) __attribute__((unused));
static void fftwf_ifft_1d_inplace(std::complex<float>* data, int n) {
    ::fftwf_plan_s* plan;
    {
        FFTW_PLAN_LOCK();
        plan = ::fftwf_plan_dft_1d(
            n,
            reinterpret_cast< ::fftwf_complex*>(data),
            reinterpret_cast< ::fftwf_complex*>(data),
            FFTW_BACKWARD,
            FFTW_ESTIMATE
        );
    }
    ::fftwf_execute(plan);
    {
        FFTW_PLAN_LOCK();
        ::fftwf_destroy_plan(plan);
    }
    float inv_n = 1.0f / static_cast<float>(n);
    for (int i = 0; i < n; ++i) {
        data[i] *= inv_n;
    }
}

// In-place 1D FFT (forward) on a contiguous array of complex<float>, length n.
static void fftwf_fft_1d_inplace(std::complex<float>* data, int n) {
    ::fftwf_plan_s* plan;
    {
        FFTW_PLAN_LOCK();
        plan = ::fftwf_plan_dft_1d(
            n,
            reinterpret_cast< ::fftwf_complex*>(data),
            reinterpret_cast< ::fftwf_complex*>(data),
            FFTW_FORWARD,
            FFTW_ESTIMATE
        );
    }
    ::fftwf_execute(plan);
    {
        FFTW_PLAN_LOCK();
        ::fftwf_destroy_plan(plan);
    }
}

// Batch 1D IFFT on columns of a contiguous (n x n_cols) column-major matrix.
static void fftwf_ifft_batch_cols(std::complex<float>* data, int n, int n_cols) __attribute__((unused));
static void fftwf_ifft_batch_cols(std::complex<float>* data, int n, int n_cols) {
    int fft_dims[] = {n};
    ::fftwf_plan_s* plan;
    {
        FFTW_PLAN_LOCK();
        plan = ::fftwf_plan_many_dft(
            1, fft_dims, n_cols,
            reinterpret_cast< ::fftwf_complex*>(data), NULL, 1, n,
            reinterpret_cast< ::fftwf_complex*>(data), NULL, 1, n,
            FFTW_BACKWARD, FFTW_ESTIMATE
        );
    }
    ::fftwf_execute(plan);
    {
        FFTW_PLAN_LOCK();
        ::fftwf_destroy_plan(plan);
    }
    float inv_n = 1.0f / static_cast<float>(n);
    int total = n * n_cols;
    for (int i = 0; i < total; ++i) {
        data[i] *= inv_n;
    }
}

#endif // ARMA_USE_FFTW3


// =============================================================================
// DIMENSION-AGNOSTIC CONVOLUTION PIPELINE (LIKE MATLAB)
// =============================================================================

// Complete N-dimensional convolution: broadcast multiply + IFFT + extract grid
// This handles ANY dimension (1D, 2D, 3D, etc.) like MATLAB's code:
//   for ix = regs.dx:-1:1
//       m = ifft(m, [], ix);
//   end
//
// kdf: (L1, L2, ..., Ldx, dh) - kernel in Fourier domain
// y_ft: (L1, L2, ..., Ldx, dy) - data in Fourier domain
// L: spatial dimensions [L1, L2, ...]
// qout: extraction indices matrix (2 x dx), 1-based R indices
// use_single: if true AND FFTW3 is available, use float32 FFT (for accuracy<=4)
// Returns: (N1, N2, ..., Ndx, dh, dy) - convolution result
//
// [[Rcpp::export]]
ComplexVector rcpp_conv_nd_full(ComplexVector kdf, ComplexVector y_ft,
                                 IntegerVector L_vec, int dh, int dy,
                                 IntegerMatrix qout, bool y_isreal = true,
                                 bool use_single = false) {
    int dx = L_vec.size();
    
    // Compute total spatial size
    size_t L_total = 1;
    std::vector<int> L(dx), N(dx), q_start(dx);
    for (int d = 0; d < dx; ++d) {
        L[d] = L_vec[d];
        L_total *= L[d];
        q_start[d] = qout(0, d) - 1;  // Convert to 0-based
        N[d] = qout(1, d) - qout(0, d) + 1;
    }
    
    // Compute output size
    size_t N_total = 1;
    for (int d = 0; d < dx; ++d) N_total *= N[d];
    
    // Validate input sizes
    if ((size_t)kdf.size() != L_total * dh) {
        stop("kdf size mismatch: expected %d, got %d", (int)(L_total * dh), (int)kdf.size());
    }
    if ((size_t)y_ft.size() != L_total * dy) {
        stop("y_ft size mismatch: expected %d, got %d", (int)(L_total * dy), (int)y_ft.size());
    }
    
    // Compute strides for L array (column-major)
    std::vector<size_t> L_strides(dx);
    L_strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        L_strides[d] = L_strides[d-1] * L[d-1];
    }
    
    // Compute strides for N array (column-major)  
    std::vector<size_t> N_strides(dx);
    N_strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        N_strides[d] = N_strides[d-1] * N[d-1];
    }
    
    // Output array
    size_t out_total = N_total * dh * dy;
    ComplexVector result(out_total);

#ifdef ARMA_USE_FFTW3
    // ================================================================
    // SINGLE-PRECISION PATH: Pre-create FFTW plans, then run in parallel
    // ================================================================
    if (use_single) {
        // Pre-create one plan per axis dimension (outside OMP parallel region).
        // Use FFTW_ESTIMATE so plan creation doesn't touch the data arrays.
        // Then use fftwf_execute_dft() inside threads with thread-local data.
        int max_axis_len = *std::max_element(L.begin(), L.end());
        std::vector<std::complex<float>> plan_buf(max_axis_len);

        // Plans for strided axes (1D IFFT per axis)
        std::vector< ::fftwf_plan_s*> ifft_plans(dx, nullptr);
        for (int axis = 0; axis < dx; ++axis) {
            int n_axis = L[axis];
            ifft_plans[axis] = ::fftwf_plan_dft_1d(
                n_axis,
                reinterpret_cast< ::fftwf_complex*>(plan_buf.data()),
                reinterpret_cast< ::fftwf_complex*>(plan_buf.data()),
                FFTW_BACKWARD,
                FFTW_ESTIMATE
            );
        }

        // Plan for axis 0 batch IFFT (contiguous columns)
        size_t n_transforms_ax0 = L_total / L[0];
        int fft_dims[] = {L[0]};
        ::fftwf_plan_s* batch_plan = ::fftwf_plan_many_dft(
            1, fft_dims, (int)n_transforms_ax0,
            reinterpret_cast< ::fftwf_complex*>(plan_buf.data()), NULL, 1, L[0],
            reinterpret_cast< ::fftwf_complex*>(plan_buf.data()), NULL, 1, L[0],
            FFTW_BACKWARD, FFTW_ESTIMATE
        );

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(dynamic)
        #endif
        for (int d_idx = 0; d_idx < dy; ++d_idx) {
            for (int h_idx = 0; h_idx < dh; ++h_idx) {
                // Thread-local workspace in single precision
                std::vector<std::complex<float>> m_ft_f(L_total);
                std::vector<std::complex<float>> slice_f(max_axis_len);

                // Step 1: Broadcast multiply (downcast on the fly)
                for (size_t s = 0; s < L_total; ++s) {
                    size_t kdf_idx = s + h_idx * L_total;
                    size_t y_idx = s + d_idx * L_total;
                    std::complex<float> k((float)kdf[kdf_idx].r, (float)kdf[kdf_idx].i);
                    std::complex<float> y_val((float)y_ft[y_idx].r, (float)y_ft[y_idx].i);
                    m_ft_f[s] = k * y_val;
                }

                // Step 2: Inverse FFT along each dimension
                for (int axis = dx - 1; axis >= 0; --axis) {
                    int n_axis = L[axis];
                    size_t stride = L_strides[axis];
                    size_t n_transforms = L_total / n_axis;
                    float inv_n = 1.0f / static_cast<float>(n_axis);

                    if (axis == 0) {
                        // Batch IFFT on contiguous columns using pre-created plan
                        ::fftwf_execute_dft(batch_plan,
                            reinterpret_cast< ::fftwf_complex*>(m_ft_f.data()),
                            reinterpret_cast< ::fftwf_complex*>(m_ft_f.data()));
                        // Normalize
                        for (size_t i = 0; i < L_total; ++i) {
                            m_ft_f[i] *= inv_n;
                        }
                    } else {
                        // Strided axis
                        for (size_t t = 0; t < n_transforms; ++t) {
                            size_t base = 0;
                            size_t temp = t;
                            for (int d = 0; d < dx; ++d) {
                                if (d != axis) {
                                    int coord = temp % L[d];
                                    temp /= L[d];
                                    base += coord * L_strides[d];
                                }
                            }
                            for (int i = 0; i < n_axis; ++i) {
                                slice_f[i] = m_ft_f[base + i * stride];
                            }
                            ::fftwf_execute_dft(ifft_plans[axis],
                                reinterpret_cast< ::fftwf_complex*>(slice_f.data()),
                                reinterpret_cast< ::fftwf_complex*>(slice_f.data()));
                            for (int i = 0; i < n_axis; ++i) {
                                m_ft_f[base + i * stride] = slice_f[i] * inv_n;
                            }
                        }
                    }
                }

                // Step 3: Extract evaluation grid (upcast to double)
                for (size_t out_s = 0; out_s < N_total; ++out_s) {
                    size_t temp = out_s;
                    size_t in_idx = 0;
                    for (int d = 0; d < dx; ++d) {
                        int coord = temp % N[d];
                        temp /= N[d];
                        in_idx += (q_start[d] + coord) * L_strides[d];
                    }
                    size_t out_idx = out_s + h_idx * N_total + d_idx * N_total * dh;
                    if (y_isreal) {
                        result[out_idx] = Rcomplex{{(double)m_ft_f[in_idx].real(), 0.0}};
                    } else {
                        result[out_idx] = Rcomplex{{(double)m_ft_f[in_idx].real(), (double)m_ft_f[in_idx].imag()}};
                    }
                }
            }
        }

        // Clean up plans
        for (int axis = 0; axis < dx; ++axis) {
            if (ifft_plans[axis]) ::fftwf_destroy_plan(ifft_plans[axis]);
        }
        if (batch_plan) ::fftwf_destroy_plan(batch_plan);

    } else
#endif // ARMA_USE_FFTW3
    {
    // ================================================================
    // DOUBLE-PRECISION PATH (default, complex128 via Armadillo/FFTW3)
    // ================================================================
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int d_idx = 0; d_idx < dy; ++d_idx) {
        for (int h_idx = 0; h_idx < dh; ++h_idx) {
            std::vector<std::complex<double>> m_ft(L_total);

            // Step 1: Broadcast multiply
            // m_ft[spatial] = kdf[spatial, h] * y_ft[spatial, d]
            for (size_t s = 0; s < L_total; ++s) {
                size_t kdf_idx = s + h_idx * L_total;
                size_t y_idx = s + d_idx * L_total;
                std::complex<double> k(kdf[kdf_idx].r, kdf[kdf_idx].i);
                std::complex<double> y(y_ft[y_idx].r, y_ft[y_idx].i);
                m_ft[s] = k * y;
            }

            // Step 2: Inverse FFT along each dimension (like MATLAB)
            // for ix = dx:-1:1; m = ifft(m, [], ix); end
            // OPTIMIZATION: Pre-allocate slice and result once per thread
            int max_axis_len = *std::max_element(L.begin(), L.end());
            cx_vec slice(max_axis_len);
            cx_vec ifft_result(max_axis_len);

            for (int axis = dx - 1; axis >= 0; --axis) {
                int n_axis = L[axis];
                size_t stride = L_strides[axis];
                size_t n_transforms = L_total / n_axis;

                // Resize only if needed (avoids allocation)
                if ((int)slice.n_elem != n_axis) {
                    slice.set_size(n_axis);
                    ifft_result.set_size(n_axis);
                }

                // OPTIMIZATION: For axis 0 (stride=1), data is contiguous
                // Can use Armadillo's matrix FFT for better performance
                if (axis == 0) {
                    // Process all transforms at once using matrix FFT
                    // Reshape m_ft as (n_axis x n_transforms) matrix
                    cx_mat m_mat(m_ft.data(), n_axis, n_transforms, false, true);
                    // Apply IFFT to each column (in-place not possible, but fewer allocations)
                    cx_mat ifft_mat = arma::ifft(m_mat);
                    // Copy back (in-place)
                    std::memcpy(m_ft.data(), ifft_mat.memptr(), L_total * sizeof(std::complex<double>));
                } else {
                    // Strided axis - use loop with pre-allocated vectors
                    for (size_t t = 0; t < n_transforms; ++t) {
                        // Compute base index for this transform
                        size_t base = 0;
                        size_t temp = t;
                        for (int d = 0; d < dx; ++d) {
                            if (d != axis) {
                                int coord = temp % L[d];
                                temp /= L[d];
                                base += coord * L_strides[d];
                            }
                        }

                        // Extract slice along axis (strided access)
                        for (int i = 0; i < n_axis; ++i) {
                            slice(i) = m_ft[base + i * stride];
                        }

                        // IFFT (reuse ifft_result vector)
                        ifft_result = arma::ifft(slice);

                        // Store back (strided)
                        for (int i = 0; i < n_axis; ++i) {
                            m_ft[base + i * stride] = ifft_result(i);
                        }
                    }
                }
            }
            
            // Step 3: Extract evaluation grid
            // Iterate over output grid points
            for (size_t out_s = 0; out_s < N_total; ++out_s) {
                // Convert output linear index to coordinates
                std::vector<int> out_coords(dx);
                size_t temp = out_s;
                for (int d = 0; d < dx; ++d) {
                    out_coords[d] = temp % N[d];
                    temp /= N[d];
                }
                
                // Map to input coordinates (add offset)
                size_t in_idx = 0;
                for (int d = 0; d < dx; ++d) {
                    in_idx += (q_start[d] + out_coords[d]) * L_strides[d];
                }
                
                // Output index: (spatial, h, d) in column-major
                size_t out_idx = out_s + h_idx * N_total + d_idx * N_total * dh;
                
                if (y_isreal) {
                    result[out_idx] = Rcomplex{{m_ft[in_idx].real(), 0.0}};
                } else {
                    result[out_idx] = Rcomplex{{m_ft[in_idx].real(), m_ft[in_idx].imag()}};
                }
            }
        }
    }
    } // end double-precision else block

    // Set output dimensions: (N1, N2, ..., Ndx, dh, dy)
    IntegerVector out_dims(dx + 2);
    for (int d = 0; d < dx; ++d) out_dims[d] = N[d];
    out_dims[dx] = dh;
    out_dims[dx + 1] = dy;
    result.attr("dim") = out_dims;
    
    return result;
}


// =============================================================================
// GAUSSIAN KERNEL FOR GRID (ELIMINATES ARRAY/APERM OVERHEAD)
// =============================================================================

// Compute Gaussian kernel on N-dimensional grid for multiple bandwidths
// xgrid: (N1, N2, ..., Ndx, dx) - grid coordinates
// h: (dh, dx) - bandwidth matrix
// Returns: (N1, N2, ..., Ndx, dh) - kernel values
//
// This eliminates the expensive array/aperm operations in R's kernel_function
// [[Rcpp::export]]
NumericVector rcpp_gaussian_kernel_grid(NumericVector xgrid, NumericMatrix h) {
    IntegerVector xgrid_dims = xgrid.attr("dim");
    int ndim = xgrid_dims.size();
    int dx = xgrid_dims[ndim - 1];  // Last dimension is dx
    int dh = h.nrow();
    
    if (h.ncol() != dx) {
        stop("h must have dx columns");
    }
    
    // Compute spatial grid size
    size_t n_spatial = 1;
    std::vector<int> N(ndim - 1);
    for (int d = 0; d < ndim - 1; ++d) {
        N[d] = xgrid_dims[d];
        n_spatial *= N[d];
    }
    
    // Output dimensions: (N1, N2, ..., dh)
    IntegerVector out_dims(ndim);  // Same as xgrid but with dh instead of dx
    for (int d = 0; d < ndim - 1; ++d) out_dims[d] = N[d];
    out_dims[ndim - 1] = dh;
    
    size_t out_total = n_spatial * dh;
    NumericVector result(out_total);
    
    // Compute determinant of bandwidth for each h
    std::vector<double> deth(dh);
    for (int ih = 0; ih < dh; ++ih) {
        deth[ih] = 1.0;
        for (int d = 0; d < dx; ++d) {
            deth[ih] *= h(ih, d);
        }
    }
    
    // Match R's kernel_function: (1/sqrt(2*pi)) * exp(-0.5 * sum_sq) / det(h)
    // Note: R uses just 1/sqrt(2*pi) regardless of dx (not raised to dx power)
    double norm_const = 1.0 / std::sqrt(2.0 * M_PI);
    
    // Process each bandwidth in parallel
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int ih = 0; ih < dh; ++ih) {
        double inv_deth = norm_const / deth[ih];
        
        // For each spatial point
        for (size_t s = 0; s < n_spatial; ++s) {
            // Compute sum of (x_d / h_d)^2 for all dimensions
            double sum_sq = 0.0;
            for (int d = 0; d < dx; ++d) {
                // xgrid index: s + d * n_spatial (column-major)
                double x_d = xgrid[s + d * n_spatial];
                double h_d = h(ih, d);
                sum_sq += (x_d / h_d) * (x_d / h_d);
            }
            
            // Gaussian kernel
            result[s + ih * n_spatial] = inv_deth * std::exp(-0.5 * sum_sq);
        }
    }
    
    result.attr("dim") = out_dims;
    return result;
}


// =============================================================================
// NUFFT TYPE-1: NON-UNIFORM TO UNIFORM (KEY FOR LARGE-N TESTS)
// =============================================================================

// Simplified NUFFT Type-1: Scattered points to uniform grid
// knot: (M, dx) - non-uniform sample positions (PRE-SCALED to [0, 2*pi] by R)
// y: (M, dy) - data values at sample points
// L: output grid size per dimension
// accuracy: NUFFT accuracy (number of digits)
// Returns: (L, L, ..., dy) - Fourier coefficients on uniform grid
//
// This implements the Gaussian gridding approach from the original fastLPR
// IMPORTANT: knots must be pre-scaled by R's scale_knots() to [0, 2*pi] range
// [[Rcpp::export]]
ComplexVector rcpp_nufft_type1(NumericMatrix knot, NumericMatrix y,
                                IntegerVector L_vec, int accuracy) {
    int M = knot.nrow();
    int dx = knot.ncol();
    int dy = y.ncol();

    if (y.nrow() != M) {
        stop("knot and y must have same number of rows");
    }
    if (L_vec.size() != dx) {
        stop("L must have length dx");
    }

    // Compute grid parameters (FIXED to match MATLAB/Pure R nufftn_type1)
    // ratio: oversampling ratio (2 for accuracy 6-11)
    // Mr = ratio * L: oversampled grid
    // Msp = ceil(accuracy): spreading width
    // tau = pi * Msp / (L^2 * ratio * (ratio - 0.5)): Gaussian width (CRITICAL FIX!)
    // hx = 2 * pi / Mr: grid spacing in angular coordinates (CRITICAL FIX!)

    // Determine oversampling ratio based on accuracy (matches compute_grid_params.R)
    double ratio;
    if (accuracy <= 6) {
        ratio = 1.0;  // No oversampling
    } else if (accuracy <= 11) {
        ratio = 2.0;  // 2x oversampling (standard)
    } else {
        ratio = 3.0;  // 3x oversampling
    }

    int Msp = (int)std::ceil(accuracy);
    int Msp2 = 2 * Msp + 1;  // Number of spreading points per dimension

    // Compute total grid sizes
    std::vector<int> L(dx), Mr(dx);
    size_t L_total = 1, Mr_total = 1;
    std::vector<double> tau(dx), hx(dx);

    for (int d = 0; d < dx; ++d) {
        L[d] = L_vec[d];
        Mr[d] = (int)(ratio * L[d]);  // Oversampled grid = ratio * L
        L_total *= L[d];
        Mr_total *= Mr[d];
        // CRITICAL FIX: Grid spacing in angular coordinates
        hx[d] = 2.0 * M_PI / Mr[d];
        // CRITICAL FIX: Gaussian width with proper denominator
        tau[d] = (M_PI * Msp) / (L[d] * L[d] * ratio * (ratio - 0.5));
    }
    
    // Allocate oversampled grid (Mr[0] x Mr[1] x ... x dy)
    std::vector<std::complex<double>> Ftau(Mr_total * dy, std::complex<double>(0, 0));
    
    // Precompute spreading grid points
    std::vector<int> spread_offsets(Msp2);
    for (int i = 0; i < Msp2; ++i) {
        spread_offsets[i] = i - Msp;  // -Msp to +Msp
    }
    
    // Compute Mr strides for column-major indexing
    std::vector<size_t> Mr_strides(dx);
    Mr_strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        Mr_strides[d] = Mr_strides[d-1] * Mr[d-1];
    }
    
    // Process each data point (parallel over data points)
    // Use critical section for accumulation
    // NOTE: knots are PRE-SCALED to [0, 2*pi] by R's scale_knots() function
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        // Thread-local accumulation buffer
        std::vector<std::complex<double>> local_Ftau(Mr_total * dy, std::complex<double>(0, 0));

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic)
        #endif
        for (int m = 0; m < M; ++m) {
            // Get position for this sample (already in [0, 2*pi] from scale_knots)
            std::vector<double> xmod(dx);
            std::vector<int> base_idx(dx);
            for (int d = 0; d < dx; ++d) {
                // knot is already scaled to [0, 2*pi] by R
                xmod[d] = knot(m, d);
                // Compute base grid index: round(xmod / hx)
                base_idx[d] = (int)std::round(xmod[d] / hx[d]);
            }

            // Compute spreading kernel for all dx dimensions
            // Total spreading points: Msp2^dx
            int n_spread = 1;
            for (int d = 0; d < dx; ++d) n_spread *= Msp2;

            for (int spread_idx = 0; spread_idx < n_spread; ++spread_idx) {
                // Convert spread_idx to multi-dimensional offset
                std::vector<int> offsets(dx);
                int temp = spread_idx;
                for (int d = 0; d < dx; ++d) {
                    offsets[d] = spread_offsets[temp % Msp2];
                    temp /= Msp2;
                }

                // Compute grid index with wrapping
                size_t grid_idx = 0;
                double weight = 1.0;

                for (int d = 0; d < dx; ++d) {
                    int idx = base_idx[d] + offsets[d];
                    // Wrap to [0, Mr[d])
                    idx = ((idx % Mr[d]) + Mr[d]) % Mr[d];
                    grid_idx += idx * Mr_strides[d];

                    // CRITICAL FIX: Compute heat kernel weight using angular coordinates
                    // diff = xmod - hx * (base_idx + offset)
                    // This matches Pure R: diff <- xmod - hx * mpmm_reshaped
                    double diff = xmod[d] - hx[d] * (base_idx[d] + offsets[d]);
                    weight *= std::exp(-(diff * diff) / (4.0 * tau[d]));
                }

                // Accumulate weighted data
                for (int d_y = 0; d_y < dy; ++d_y) {
                    size_t out_idx = grid_idx + d_y * Mr_total;
                    // Handle complex y (imaginary part is 0 for real data)
                    local_Ftau[out_idx] += std::complex<double>(y(m, d_y) * weight, 0);
                }
            }
        }

        // Merge thread-local results
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            for (size_t i = 0; i < Mr_total * dy; ++i) {
                Ftau[i] += local_Ftau[i];
            }
        }
    }

    // Apply FFT to oversampled grid with fftshift
    // This matches pure R nufftn_type1: fftshift_array(apply_fft_axis(Ftau, axis, FALSE), axis)
    // For each response variable, apply dx-dimensional FFT

#ifdef ARMA_USE_FFTW3
    // Single-precision FFT path for accuracy <= 4
    if (accuracy <= 4) {
        std::vector<std::complex<float>> Ftau_f(Mr_total * dy);
        for (size_t i = 0; i < Mr_total * (size_t)dy; ++i) {
            Ftau_f[i] = std::complex<float>((float)Ftau[i].real(), (float)Ftau[i].imag());
        }
        std::vector<std::complex<float>> slice_f;
        for (int d_y = 0; d_y < dy; ++d_y) {
            for (int d = dx - 1; d >= 0; --d) {
                int n_axis = Mr[d];
                size_t stride = Mr_strides[d];
                size_t n_transforms = Mr_total / n_axis;
                int shift = (n_axis + 1) / 2;
                slice_f.resize(n_axis);
                for (size_t t = 0; t < n_transforms; ++t) {
                    size_t base = d_y * Mr_total;
                    size_t temp = t;
                    for (int dd = 0; dd < dx; ++dd) {
                        if (dd != d) {
                            int coord = temp % Mr[dd];
                            temp /= Mr[dd];
                            base += coord * Mr_strides[dd];
                        }
                    }
                    for (int i = 0; i < n_axis; ++i) {
                        slice_f[i] = Ftau_f[base + i * stride];
                    }
                    fftwf_fft_1d_inplace(slice_f.data(), n_axis);
                    for (int i = 0; i < n_axis; ++i) {
                        int src_idx = (i + shift) % n_axis;
                        Ftau_f[base + i * stride] = slice_f[src_idx];
                    }
                }
            }
        }
        for (size_t i = 0; i < Mr_total * (size_t)dy; ++i) {
            Ftau[i] = std::complex<double>((double)Ftau_f[i].real(), (double)Ftau_f[i].imag());
        }
    } else
#endif
    {
        for (int d_y = 0; d_y < dy; ++d_y) {
            for (int d = dx - 1; d >= 0; --d) {
                int n_axis = Mr[d];
                size_t stride = Mr_strides[d];
                size_t n_transforms = Mr_total / n_axis;
                int shift = (n_axis + 1) / 2;
                cx_vec slice(n_axis);
                for (size_t t = 0; t < n_transforms; ++t) {
                    size_t base = d_y * Mr_total;
                    size_t temp = t;
                    for (int dd = 0; dd < dx; ++dd) {
                        if (dd != d) {
                            int coord = temp % Mr[dd];
                            temp /= Mr[dd];
                            base += coord * Mr_strides[dd];
                        }
                    }
                    for (int i = 0; i < n_axis; ++i) {
                        slice(i) = Ftau[base + i * stride];
                    }
                    cx_vec fft_result = arma::fft(slice);
                    for (int i = 0; i < n_axis; ++i) {
                        int src_idx = (i + shift) % n_axis;
                        Ftau[base + i * stride] = fft_result(src_idx);
                    }
                }
            }
        }
    }

    // Normalize by M * ratio^dx (matching pure R: Ftau <- Ftau / (M * (ratio^dx)))
    // Use the computed ratio variable from grid params, not hardcoded 2.0
    double norm_factor = M * std::pow(ratio, dx);
    for (size_t i = 0; i < Mr_total * dy; ++i) {
        Ftau[i] /= norm_factor;
    }

    // Extract center portion (L from Mr)
    // q = (Mr - L) / 2, extract [q+1, q+L]
    size_t out_total = L_total * dy;
    ComplexVector result(out_total);

    std::vector<size_t> L_strides(dx);
    L_strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        L_strides[d] = L_strides[d-1] * L[d-1];
    }

    std::vector<int> q(dx);
    for (int d = 0; d < dx; ++d) {
        q[d] = (Mr[d] - L[d]) / 2;
    }

    for (int d_y = 0; d_y < dy; ++d_y) {
        for (size_t out_s = 0; out_s < L_total; ++out_s) {
            // Convert to multi-dimensional index
            std::vector<int> out_coords(dx);
            size_t temp = out_s;
            for (int d = 0; d < dx; ++d) {
                out_coords[d] = temp % L[d];
                temp /= L[d];
            }

            // Map to Mr grid
            size_t in_idx = d_y * Mr_total;
            for (int d = 0; d < dx; ++d) {
                int mr_coord = q[d] + out_coords[d];
                in_idx += mr_coord * Mr_strides[d];
            }

            size_t out_idx = out_s + d_y * L_total;
            result[out_idx] = Rcomplex{{Ftau[in_idx].real(), Ftau[in_idx].imag()}};
        }
    }

    // Set output dimensions
    IntegerVector out_dims(dx + 1);
    for (int d = 0; d < dx; ++d) out_dims[d] = L[d];
    out_dims[dx] = dy;
    result.attr("dim") = out_dims;

    return result;
}


// =============================================================================
// NUFFT TYPE-1 FOR COMPLEX-VALUED DATA
// =============================================================================

// NUFFT Type-1 for complex-valued y data
// knot: (M, dx) - non-uniform sample positions (PRE-SCALED to [0, 2*pi] by R)
// y: (M, dy) - COMPLEX data values at sample points
// L: output grid size per dimension
// accuracy: NUFFT accuracy (number of digits)
// Returns: (L, L, ..., dy) - Complex Fourier coefficients on uniform grid
//
// This is identical to rcpp_nufft_type1 but accepts ComplexMatrix y
// IMPORTANT: knots must be pre-scaled by R's scale_knots() to [0, 2*pi] range
// [[Rcpp::export]]
ComplexVector rcpp_nufft_type1_complex(NumericMatrix knot, ComplexMatrix y,
                                        IntegerVector L_vec, int accuracy) {
    int M = knot.nrow();
    int dx = knot.ncol();
    int dy = y.ncol();

    if (y.nrow() != M) {
        stop("knot and y must have same number of rows");
    }
    if (L_vec.size() != dx) {
        stop("L must have length dx");
    }

    // Compute grid parameters (FIXED to match MATLAB/Pure R nufftn_type1)
    // ratio: oversampling ratio (2 for accuracy 6-11)
    // Mr = ratio * L: oversampled grid
    // Msp = ceil(accuracy): spreading width
    // tau = pi * Msp / (L^2 * ratio * (ratio - 0.5)): Gaussian width (CRITICAL FIX!)
    // hx = 2 * pi / Mr: grid spacing in angular coordinates (CRITICAL FIX!)

    // Determine oversampling ratio based on accuracy (matches compute_grid_params.R)
    double ratio;
    if (accuracy <= 6) {
        ratio = 1.0;  // No oversampling
    } else if (accuracy <= 11) {
        ratio = 2.0;  // 2x oversampling (standard)
    } else {
        ratio = 3.0;  // 3x oversampling
    }

    int Msp = (int)std::ceil(accuracy);
    int Msp2 = 2 * Msp + 1;  // Number of spreading points per dimension

    // Compute total grid sizes
    std::vector<int> L(dx), Mr(dx);
    size_t L_total = 1, Mr_total = 1;
    std::vector<double> tau(dx), hx(dx);

    for (int d = 0; d < dx; ++d) {
        L[d] = L_vec[d];
        Mr[d] = (int)(ratio * L[d]);  // Oversampled grid = ratio * L
        L_total *= L[d];
        Mr_total *= Mr[d];
        // CRITICAL FIX: Grid spacing in angular coordinates
        hx[d] = 2.0 * M_PI / Mr[d];
        // CRITICAL FIX: Gaussian width with proper denominator
        tau[d] = (M_PI * Msp) / (L[d] * L[d] * ratio * (ratio - 0.5));
    }

    // Allocate oversampled grid (Mr[0] x Mr[1] x ... x dy) - COMPLEX
    std::vector<std::complex<double>> Ftau(Mr_total * dy, std::complex<double>(0, 0));

    // Precompute spreading grid points
    std::vector<int> spread_offsets(Msp2);
    for (int i = 0; i < Msp2; ++i) {
        spread_offsets[i] = i - Msp;  // -Msp to +Msp
    }

    // Compute Mr strides for column-major indexing
    std::vector<size_t> Mr_strides(dx);
    Mr_strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        Mr_strides[d] = Mr_strides[d-1] * Mr[d-1];
    }

    // Process each data point (parallel over data points)
    // Use critical section for accumulation
    // NOTE: knots are PRE-SCALED to [0, 2*pi] by R's scale_knots() function
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        // Thread-local accumulation buffer
        std::vector<std::complex<double>> local_Ftau(Mr_total * dy, std::complex<double>(0, 0));

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic)
        #endif
        for (int m = 0; m < M; ++m) {
            // Get position for this sample (already in [0, 2*pi] from scale_knots)
            std::vector<double> xmod(dx);
            std::vector<int> base_idx(dx);
            for (int d = 0; d < dx; ++d) {
                // knot is already scaled to [0, 2*pi] by R
                xmod[d] = knot(m, d);
                // Compute base grid index: round(xmod / hx)
                base_idx[d] = (int)std::round(xmod[d] / hx[d]);
            }

            // Compute spreading kernel for all dx dimensions
            // Total spreading points: Msp2^dx
            int n_spread = 1;
            for (int d = 0; d < dx; ++d) n_spread *= Msp2;

            for (int spread_idx = 0; spread_idx < n_spread; ++spread_idx) {
                // Convert spread_idx to multi-dimensional offset
                std::vector<int> offsets(dx);
                int temp = spread_idx;
                for (int d = 0; d < dx; ++d) {
                    offsets[d] = spread_offsets[temp % Msp2];
                    temp /= Msp2;
                }

                // Compute grid index with wrapping
                size_t grid_idx = 0;
                double weight = 1.0;

                for (int d = 0; d < dx; ++d) {
                    int idx = base_idx[d] + offsets[d];
                    // Wrap to [0, Mr[d])
                    idx = ((idx % Mr[d]) + Mr[d]) % Mr[d];
                    grid_idx += idx * Mr_strides[d];

                    // CRITICAL FIX: Compute heat kernel weight using angular coordinates
                    // diff = xmod - hx * (base_idx + offset)
                    // This matches Pure R: diff <- xmod - hx * mpmm_reshaped
                    double diff = xmod[d] - hx[d] * (base_idx[d] + offsets[d]);
                    weight *= std::exp(-(diff * diff) / (4.0 * tau[d]));
                }

                // Accumulate weighted COMPLEX data
                for (int d_y = 0; d_y < dy; ++d_y) {
                    size_t out_idx = grid_idx + d_y * Mr_total;
                    // Extract complex value from R's Rcomplex struct
                    Rcomplex y_val = y(m, d_y);
                    local_Ftau[out_idx] += std::complex<double>(y_val.r * weight, y_val.i * weight);
                }
            }
        }

        // Merge thread-local results
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            for (size_t i = 0; i < Mr_total * dy; ++i) {
                Ftau[i] += local_Ftau[i];
            }
        }
    }

    // Apply FFT to oversampled grid with fftshift
    // This matches pure R nufftn_type1: fftshift_array(apply_fft_axis(Ftau, axis, FALSE), axis)
    // For each response variable, apply dx-dimensional FFT
    for (int d_y = 0; d_y < dy; ++d_y) {
        // Apply FFT along each dimension, then fftshift
        for (int d = dx - 1; d >= 0; --d) {  // Process in reverse order like R: for (ix in dx:1)
            int n_axis = Mr[d];
            size_t stride = Mr_strides[d];
            size_t n_transforms = Mr_total / n_axis;
            int shift = (n_axis + 1) / 2;  // fftshift amount: ceiling(n/2) - FIXED from floor(n/2)

            cx_vec slice(n_axis);

            for (size_t t = 0; t < n_transforms; ++t) {
                // Compute base index
                size_t base = d_y * Mr_total;
                size_t temp = t;
                for (int dd = 0; dd < dx; ++dd) {
                    if (dd != d) {
                        int coord = temp % Mr[dd];
                        temp /= Mr[dd];
                        base += coord * Mr_strides[dd];
                    }
                }

                // Extract slice
                for (int i = 0; i < n_axis; ++i) {
                    slice(i) = Ftau[base + i * stride];
                }

                // FFT (forward)
                cx_vec fft_result = arma::fft(slice);

                // Apply fftshift: [shift+1:n, 1:shift] for 1-indexed
                // In 0-indexed: [shift:n-1, 0:shift-1]
                // Store back with fftshift applied
                for (int i = 0; i < n_axis; ++i) {
                    int src_idx = (i + shift) % n_axis;  // fftshift mapping
                    Ftau[base + i * stride] = fft_result(src_idx);
                }
            }
        }
    }

    // Normalize by M * ratio^dx (matching pure R: Ftau <- Ftau / (M * (ratio^dx)))
    // Use the computed ratio variable from grid params, not hardcoded 2.0
    double norm_factor_complex = M * std::pow(ratio, dx);
    for (size_t i = 0; i < Mr_total * dy; ++i) {
        Ftau[i] /= norm_factor_complex;
    }

    // Extract center portion (L from Mr)
    // q = (Mr - L) / 2, extract [q+1, q+L]
    size_t out_total = L_total * dy;
    ComplexVector result(out_total);

    std::vector<size_t> L_strides(dx);
    L_strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        L_strides[d] = L_strides[d-1] * L[d-1];
    }

    std::vector<int> q(dx);
    for (int d = 0; d < dx; ++d) {
        q[d] = (Mr[d] - L[d]) / 2;
    }

    for (int d_y = 0; d_y < dy; ++d_y) {
        for (size_t out_s = 0; out_s < L_total; ++out_s) {
            // Convert to multi-dimensional index
            std::vector<int> out_coords(dx);
            size_t temp = out_s;
            for (int d = 0; d < dx; ++d) {
                out_coords[d] = temp % L[d];
                temp /= L[d];
            }

            // Map to Mr grid
            size_t in_idx = d_y * Mr_total;
            for (int d = 0; d < dx; ++d) {
                int mr_coord = q[d] + out_coords[d];
                in_idx += mr_coord * Mr_strides[d];
            }

            size_t out_idx = out_s + d_y * L_total;
            result[out_idx] = Rcomplex{{Ftau[in_idx].real(), Ftau[in_idx].imag()}};
        }
    }

    // Set output dimensions
    IntegerVector out_dims(dx + 1);
    for (int d = 0; d < dx; ++d) out_dims[d] = L[d];
    out_dims[dx] = dy;
    result.attr("dim") = out_dims;

    return result;
}


// =============================================================================
// SCALE KNOTS FOR NUFFT
// =============================================================================
//
// Fast C++ implementation of scale_knots for NUFFT preprocessing.
// Scales knot locations to [0, 2*pi] range required by Greengard's NUFFT.
//
// This is 5-10x faster than the pure R implementation for large N.
//
// knot: (M, dx) - original knot locations
// N: (dx,) - grid size per dimension
// Fs: (dx,) - sampling frequency per dimension
// Returns: (M, dx) - scaled knot locations in [0, 2*pi]
//
// [[Rcpp::export]]
NumericMatrix rcpp_scale_knots(NumericMatrix knot, NumericVector N, NumericVector Fs) {
    int M = knot.nrow();
    int dx = knot.ncol();
    
    if (N.size() != dx || Fs.size() != dx) {
        stop("N and Fs must have length dx");
    }
    
    // Allocate output
    NumericMatrix result(M, dx);
    
    // Process each dimension
    for (int d = 0; d < dx; ++d) {
        double scale = Fs[d] / N[d];
        
        // First pass: scale and find min/max
        double kmin = knot(0, d) * scale;
        double kmax = kmin;
        
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            double local_min = kmin;
            double local_max = kmax;
            
            #ifdef _OPENMP
            #pragma omp for nowait
            #endif
            for (int i = 0; i < M; ++i) {
                double val = knot(i, d) * scale;
                result(i, d) = val;
                if (val < local_min) local_min = val;
                if (val > local_max) local_max = val;
            }
            
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                if (local_min < kmin) kmin = local_min;
                if (local_max > kmax) kmax = local_max;
            }
        }
        
        // Compute shift
        double center = (kmin + kmax) / 2.0 - 0.5;
        double shift = -0.5 - center;
        
        // Second pass: shift and convert to [0, 2*pi]
        const double TWO_PI = 2.0 * M_PI;
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < M; ++i) {
            double val = (result(i, d) + shift) * TWO_PI;
            // Modulo 2*pi (handle negative values)
            val = val - TWO_PI * std::floor(val / TWO_PI);
            result(i, d) = val;
        }
    }
    
    return result;
}


// =============================================================================
// NUFFT TYPE-1 BINNING MODE (ACCURACY=0)
// =============================================================================
//
// Fast binning for NUFFT when accuracy=0 (no spreading, direct accumarray)
// This replaces the slow R for-loop in nufft.R lines 340-348
//
// knot: (M, dx) - non-uniform sample positions (PRE-SCALED to [0, 2*pi] by R)
// y: (M, dy) - data values at sample points (real)
// Mr: grid size per dimension (oversampled)
// Returns: (Mr[0], Mr[1], ..., dy) - binned values on uniform grid
//
// [[Rcpp::export]]
ComplexVector rcpp_nufft_binning(NumericMatrix knot, NumericMatrix y,
                                  IntegerVector Mr_vec) {
    int M = knot.nrow();
    int dx = knot.ncol();
    int dy = y.ncol();

    if (y.nrow() != M) {
        stop("knot and y must have same number of rows");
    }
    if (Mr_vec.size() != dx) {
        stop("Mr must have length dx");
    }

    // Compute total grid size
    std::vector<int> Mr(dx);
    size_t Mr_total = 1;
    for (int d = 0; d < dx; ++d) {
        Mr[d] = Mr_vec[d];
        Mr_total *= Mr[d];
    }

    // Grid spacing in angular coordinates
    std::vector<double> hx(dx);
    for (int d = 0; d < dx; ++d) {
        hx[d] = 2.0 * M_PI / Mr[d];
    }

    // Compute strides for column-major indexing
    std::vector<size_t> strides(dx);
    strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        strides[d] = strides[d-1] * Mr[d-1];
    }

    // Allocate output grid (complex for FFT compatibility)
    std::vector<std::complex<double>> Ftau(Mr_total * dy, std::complex<double>(0, 0));

    // Parallel binning with thread-local accumulation
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        // Thread-local accumulation buffer
        std::vector<std::complex<double>> local_Ftau(Mr_total * dy, std::complex<double>(0, 0));

        #ifdef _OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int m = 0; m < M; ++m) {
            // Compute grid index for this sample
            size_t grid_idx = 0;
            for (int d = 0; d < dx; ++d) {
                // knot is already scaled to [0, 2*pi] by R
                int idx = (int)std::round(knot(m, d) / hx[d]);
                // Wrap to [0, Mr[d])
                idx = ((idx % Mr[d]) + Mr[d]) % Mr[d];
                grid_idx += idx * strides[d];
            }

            // Accumulate for each response variable
            for (int d_y = 0; d_y < dy; ++d_y) {
                size_t out_idx = grid_idx + d_y * Mr_total;
                local_Ftau[out_idx] += std::complex<double>(y(m, d_y), 0);
            }
        }

        // Merge thread-local results
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            for (size_t i = 0; i < Mr_total * dy; ++i) {
                Ftau[i] += local_Ftau[i];
            }
        }
    }

    // Convert to R ComplexVector
    ComplexVector result(Mr_total * dy);
    for (size_t i = 0; i < Mr_total * dy; ++i) {
        result[i] = Rcomplex{{Ftau[i].real(), Ftau[i].imag()}};
    }

    // Set output dimensions: (Mr[0], Mr[1], ..., dy)
    IntegerVector out_dims(dx + 1);
    for (int d = 0; d < dx; ++d) out_dims[d] = Mr[d];
    out_dims[dx] = dy;
    result.attr("dim") = out_dims;

    return result;
}

// Complex version of binning
// [[Rcpp::export]]
ComplexVector rcpp_nufft_binning_complex(NumericMatrix knot, ComplexMatrix y,
                                          IntegerVector Mr_vec) {
    int M = knot.nrow();
    int dx = knot.ncol();
    int dy = y.ncol();

    if (y.nrow() != M) {
        stop("knot and y must have same number of rows");
    }
    if (Mr_vec.size() != dx) {
        stop("Mr must have length dx");
    }

    // Compute total grid size
    std::vector<int> Mr(dx);
    size_t Mr_total = 1;
    for (int d = 0; d < dx; ++d) {
        Mr[d] = Mr_vec[d];
        Mr_total *= Mr[d];
    }

    // Grid spacing in angular coordinates
    std::vector<double> hx(dx);
    for (int d = 0; d < dx; ++d) {
        hx[d] = 2.0 * M_PI / Mr[d];
    }

    // Compute strides for column-major indexing
    std::vector<size_t> strides(dx);
    strides[0] = 1;
    for (int d = 1; d < dx; ++d) {
        strides[d] = strides[d-1] * Mr[d-1];
    }

    // Allocate output grid
    std::vector<std::complex<double>> Ftau(Mr_total * dy, std::complex<double>(0, 0));

    // Parallel binning with thread-local accumulation
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<std::complex<double>> local_Ftau(Mr_total * dy, std::complex<double>(0, 0));

        #ifdef _OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int m = 0; m < M; ++m) {
            size_t grid_idx = 0;
            for (int d = 0; d < dx; ++d) {
                int idx = (int)std::round(knot(m, d) / hx[d]);
                idx = ((idx % Mr[d]) + Mr[d]) % Mr[d];
                grid_idx += idx * strides[d];
            }

            for (int d_y = 0; d_y < dy; ++d_y) {
                size_t out_idx = grid_idx + d_y * Mr_total;
                Rcomplex y_val = y(m, d_y);
                local_Ftau[out_idx] += std::complex<double>(y_val.r, y_val.i);
            }
        }

        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            for (size_t i = 0; i < Mr_total * dy; ++i) {
                Ftau[i] += local_Ftau[i];
            }
        }
    }

    ComplexVector result(Mr_total * dy);
    for (size_t i = 0; i < Mr_total * dy; ++i) {
        result[i] = Rcomplex{{Ftau[i].real(), Ftau[i].imag()}};
    }

    IntegerVector out_dims(dx + 1);
    for (int d = 0; d < dx; ++d) out_dims[d] = Mr[d];
    out_dims[dx] = dy;
    result.attr("dim") = out_dims;

    return result;
}


// =============================================================================
// VERSION AND CAPABILITY CHECK
// =============================================================================

// [[Rcpp::export]]
List rcpp_info() {
    int n_threads = 1;
    bool openmp_enabled = false;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
    openmp_enabled = true;
#endif
    
    return List::create(
        Named("version") = "2.0.0",
        Named("rcpp_version") = RCPP_VERSION,
        Named("armadillo_version") = ARMA_VERSION_MAJOR * 10000 + 
                                     ARMA_VERSION_MINOR * 100 + 
                                     ARMA_VERSION_PATCH,
        Named("openmp_enabled") = openmp_enabled,
        Named("num_threads") = n_threads,
        Named("capabilities") = CharacterVector::create(
            "fft_nd", "fft2d_batch", "interp_nd", "interp_batch_1d",
            "aperm", "complex_ops", "broadcast_multiply", "openmp",
            "nufft_type1", "nufft_type1_complex"
        )
    );
}
