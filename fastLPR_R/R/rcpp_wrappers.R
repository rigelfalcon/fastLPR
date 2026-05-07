# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

# rcpp_wrappers.R - R wrappers for Rcpp-accelerated functions
#
# This module provides:
# 1. Automatic fallback to pure R if Rcpp not available
# 2. use_rcpp option for all accelerated functions
# 3. Consistent interface matching pure R functions

# Global cache environment for Rcpp availability (avoids locked binding issues)
.fastlpr_cache <- new.env(parent = emptyenv())
.fastlpr_cache$rcpp_available <- NULL
.fastlpr_cache$rcpp_error <- NULL
.fastlpr_cache$allow_pure_r <- NULL

#' Check if Rcpp acceleration is available
#'
#' @return TRUE if Rcpp functions are available, FALSE otherwise
#' @export
rcpp_available <- function() {
    if (is.null(.fastlpr_cache$rcpp_available)) {
        .fastlpr_cache$rcpp_error <- NULL
        .fastlpr_cache$rcpp_available <- tryCatch({
            rcpp_info()
            TRUE
        }, error = function(e) {
            .fastlpr_cache$rcpp_error <- conditionMessage(e)
            FALSE
        })
    }
    return(.fastlpr_cache$rcpp_available)
}

#' Check if Pure R fallback is allowed (internal)
#'
#' @return TRUE if pure R fallback is allowed via FASTLPR_ALLOW_PURE_R env var
#' @noRd
.allow_pure_r <- function() {
    env_val <- Sys.getenv("FASTLPR_ALLOW_PURE_R", unset = "")
    allow_pure_r <- (env_val == "1" || tolower(env_val) == "true")
    .fastlpr_cache$allow_pure_r <- allow_pure_r
    return(allow_pure_r)
}

get_rcpp_error <- function() {
    if (is.null(.fastlpr_cache$rcpp_available)) {
        rcpp_available()
    }
    return(.fastlpr_cache$rcpp_error)
}

#' Require Rcpp or error (for fallback points)
#'
#' Call this at Pure R fallback points. If Rcpp is not available and
#' FASTLPR_ALLOW_PURE_R is not set, throws an informative error.
#'
#' @param context Character string describing where the fallback occurred
#' @return TRUE if fallback is allowed, otherwise throws error
#' @noRd
require_rcpp_or_allow_fallback <- function(context = "unknown") {
    if (.allow_pure_r()) {
        return(TRUE)  # Pure R fallback is allowed
    }
    rcpp_error <- get_rcpp_error()
    rcpp_detail <- ""
    if (!is.null(rcpp_error) && nzchar(rcpp_error)) {
        rcpp_detail <- paste0("\nRcpp error:\n  ", rcpp_error)
    }
    stop(paste0(
        sprintf("fastlpr: Rcpp function not available in %s.\n", context),
        "Pure R fallback is disabled for performance reasons.",
        rcpp_detail,
        "\n\nOptions:\n",
        "  1. Compile Rcpp: See fastLPR_R/src/README.md\n",
        "  2. Allow pure R (SLOW): Sys.setenv(FASTLPR_ALLOW_PURE_R = '1')\n"
    ), call. = FALSE)
}

#' Get Rcpp information
#'
#' @return List with Rcpp version and capabilities, or NULL if not available
#' @export
get_rcpp_info <- function() {
    if (rcpp_available()) {
        return(rcpp_info())
    }
    return(NULL)
}

#' N-dimensional FFT with optional Rcpp acceleration
#'
#' @param arr Complex or numeric array
#' @param dims Integer vector of dimensions (optional, auto-detect if NULL)
#' @param inverse Logical, if TRUE compute inverse FFT
#' @param use_rcpp Logical, if TRUE use Rcpp (default: auto-detect)
#'
#' @return Complex array of same dimensions
#' @export
fast_fft_nd <- function(arr, dims = NULL, inverse = FALSE, use_rcpp = NULL) {
    if (is.null(dims)) {
        dims <- dim(arr)
        if (is.null(dims)) dims <- length(arr)
    }
    dims <- as.integer(dims)
    
    if (is.null(use_rcpp)) {
        use_rcpp <- get_rcpp_preference()
    }
    
    rcpp_ok <- rcpp_available()
    if (use_rcpp && rcpp_ok) {
        if (is.complex(arr)) {
            return(rcpp_fft_nd(arr, dims, inverse))
        } else {
            return(rcpp_fft_nd_real(as.numeric(arr), dims, inverse))
        }
    }
    if (!rcpp_ok) {
        require_rcpp_or_allow_fallback("fast_fft_nd")
    }
    return(fft_nd_r(arr, inverse))
}

#' Pure R implementation of N-dimensional FFT
#' @noRd
fft_nd_r <- function(arr, inverse = FALSE) {
    d <- dim(arr)
    if (is.null(d)) {
        return(if (inverse) fft(arr, inverse = TRUE) / length(arr) else fft(arr))
    }
    
    result <- arr
    for (axis in seq_along(d)) {
        result <- apply_fft_axis_r(result, axis, inverse)
    }
    return(result)
}

#' Apply FFT along one axis (pure R)
#' @noRd
apply_fft_axis_r <- function(arr, axis, inverse = FALSE) {
    d <- dim(arr)
    n <- d[axis]
    
    perm <- c(axis, seq_along(d)[-axis])
    arr_perm <- aperm(arr, perm)
    dim(arr_perm) <- c(n, prod(d) / n)
    
    if (inverse) {
        result <- mvfft(arr_perm, inverse = TRUE) / n
    } else {
        result <- mvfft(arr_perm)
    }
    
    new_dims <- c(n, d[-axis])
    dim(result) <- new_dims
    result <- aperm(result, order(perm))
    
    return(result)
}

#' N-dimensional linear interpolation with optional Rcpp acceleration
#'
#' @param grid_vals List of grid vectors for each dimension
#' @param arr Numeric array of values at grid points
#' @param points Matrix of query points (n_points x ndim)
#' @param use_rcpp Logical, if TRUE use Rcpp acceleration
#'
#' @return Numeric vector of interpolated values
#' @export
fast_interp_nd <- function(grid_vals, arr, points, use_rcpp = NULL) {
    if (is.null(use_rcpp)) {
        use_rcpp <- get_rcpp_preference()
    }
    
    rcpp_ok <- rcpp_available()
    if (use_rcpp && rcpp_ok) {
        if (is.complex(arr)) {
            return(rcpp_interp_nd_complex(grid_vals, arr, as.matrix(points)))
        } else {
            return(rcpp_interp_nd(grid_vals, as.numeric(arr), as.matrix(points)))
        }
    }
    if (!rcpp_ok) {
        require_rcpp_or_allow_fallback("fast_interp_nd")
    }
    return(interp_nd_r(grid_vals, arr, points))
}

#' Pure R implementation of N-dimensional interpolation
#' @noRd
interp_nd_r <- function(grid_vals, arr, points) {
    ndim <- length(grid_vals)
    n_points <- nrow(points)
    result <- numeric(n_points)
    
    dims <- sapply(grid_vals, length)
    strides <- cumprod(c(1, dims[-ndim]))
    
    for (p in seq_len(n_points)) {
        lo_idx <- integer(ndim)
        t_vals <- numeric(ndim)
        
        for (d in seq_len(ndim)) {
            x <- points[p, d]
            g <- grid_vals[[d]]
            idx <- findInterval(x, g)
            
            if (idx < 1) {
                lo_idx[d] <- 1
                t_vals[d] <- 0
            } else if (idx >= length(g)) {
                lo_idx[d] <- length(g) - 1
                t_vals[d] <- 1
            } else {
                lo_idx[d] <- idx
                x0 <- g[idx]
                x1 <- g[idx + 1]
                t_vals[d] <- if (x1 > x0) (x - x0) / (x1 - x0) else 0
            }
        }
        
        n_corners <- 2^ndim
        value <- 0
        
        for (corner in 0:(n_corners - 1)) {
            weight <- 1
            idx <- 1
            
            for (d in seq_len(ndim)) {
                bit <- bitwAnd(bitwShiftR(corner, d - 1), 1)
                grid_idx <- lo_idx[d] + bit
                if (grid_idx > dims[d]) grid_idx <- dims[d]
                
                idx <- idx + (grid_idx - 1) * strides[d]
                weight <- weight * if (bit) t_vals[d] else (1 - t_vals[d])
            }
            
            value <- value + weight * arr[idx]
        }
        
        result[p] <- value
    }
    
    return(result)
}

#' Fast array permutation with optional Rcpp acceleration
#'
#' @param arr Numeric or complex array
#' @param perm Integer vector specifying permutation
#' @param use_rcpp Logical, if TRUE use Rcpp acceleration
#'
#' @return Permuted array
#' @export
fast_aperm <- function(arr, perm, use_rcpp = NULL) {
    if (is.null(use_rcpp)) {
        use_rcpp <- get_rcpp_preference()
    }
    
    dims <- dim(arr)
    if (is.null(dims)) return(arr)
    
    rcpp_ok <- rcpp_available()
    if (use_rcpp && rcpp_ok) {
        if (is.complex(arr)) {
            return(rcpp_aperm_complex(arr, as.integer(dims), as.integer(perm)))
        } else {
            return(rcpp_aperm(as.numeric(arr), as.integer(dims), as.integer(perm)))
        }
    }
    if (!rcpp_ok) {
        require_rcpp_or_allow_fallback("fast_aperm")
    }
    return(aperm(arr, perm))
}

#' Set global Rcpp preference
#'
#' @param use_rcpp Logical, if TRUE prefer Rcpp acceleration globally
#' @export
set_rcpp_preference <- function(use_rcpp) {
    options(fastlpr.use_rcpp = use_rcpp)
}

#' Get global Rcpp preference
#'
#' @return Current Rcpp preference (default: TRUE if available)
#' @export
get_rcpp_preference <- function() {
    pref <- getOption("fastlpr.use_rcpp", default = NULL)
    if (is.null(pref)) {
        return(rcpp_available())
    }
    return(pref)
}
