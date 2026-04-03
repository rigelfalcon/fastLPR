# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Compute design matrix for local polynomial regression
#'
#' Computes the design matrix S = X'*W*X for local polynomial regression.
#' For order 0 (Nadaraya-Watson), S is a scalar (sum of kernel weights).
#' For order >= 1, S is a list with elements computed via convolution.
#'
#' Mirrors: fastLPR/utility/core/fastLPR_s.m
#'
#' @keywords internal
fastlpr_s <- function(regs) {

  # Create vector of ones for convolution (represents constant function)
  Is <- array(1, dim = c(regs$Tx, 1))
  storage.mode(Is) <- regs$dtype

  if (regs$opt$order == 0) {
    # Order 0: Nadaraya-Watson estimator
    # S = sum of kernel weights at each evaluation point
    # Add eps for numerical stability (prevents division by zero)
    regs$s <- fastlpr_conv(regs, regs$kdf, Is, add_eps = FALSE) + .Machine$double.eps
  } else {
    # Order >= 1: Local polynomial regression
    # Compute each element of the design matrix S via convolution
    # S has ns unique elements (lower triangular due to symmetry)
    regs$s <- list()
    for (i in regs$lwp$ns:1) {
      regs$s[[i]] <- fastlpr_conv(regs, regs$kdf[[i]], Is, add_eps = FALSE)
    }
  }

  regs
}

