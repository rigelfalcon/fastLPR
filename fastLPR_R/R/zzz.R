# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

# zzz.R - Package startup diagnostics for fastlpr
#
# This file contains package load hooks that report Rcpp acceleration status.
# Users will know at package load time whether they're getting optimized
# Rcpp performance or falling back to pure R implementations.
#
# Works both as:
# 1. Package .onLoad hook (when installed as a proper R package)
# 2. Manual startup message (when sourced directly via load_fastlpr.R)

#' @importFrom utils packageVersion
NULL

# Declare global variables to suppress R CMD check NOTEs
# These are either:
# 1. Functions dynamically looked up via get0() in fastlpr_dof.R
# 2. Legacy functions in fastlpr.R that are not yet fully implemented
# 3. Internal helper functions that are defined in the package
utils::globalVariables(c(
  # Dynamic lookups in fastlpr_dof.R
  "dof_interpolate_batch",
  "rcpp_interp_batch_1d",
  # Internal helper functions
  "interp_batch_1d",
  "interp_batch_2d",
  "interp_batch_3d",
  "sub2ind"
))

#' Generate fastlpr startup message
#'
#' Internal function to generate the startup message based on Rcpp availability.
#' If Rcpp is not available and FASTLPR_ALLOW_PURE_R is not set, throws an error.
#'
#' @return Character string with the startup message
#' @noRd
.fastlpr_startup_message <- function() {
  # Check if Rcpp acceleration is available
  rcpp_status <- tryCatch({
    info <- rcpp_info()
    list(available = TRUE, info = info, error = NULL)
  }, error = function(e) {
    list(available = FALSE, info = NULL, error = conditionMessage(e))
  })

  # Update cache if it exists
  if (exists(".fastlpr_cache", mode = "environment")) {
    .fastlpr_cache$rcpp_available <- rcpp_status$available
    .fastlpr_cache$rcpp_error <- rcpp_status$error
    .fastlpr_cache$allow_pure_r <- .allow_pure_r()
  }

  # Build startup message
  if (rcpp_status$available) {
    # Get thread count - either from rcpp_info or rcpp_get_num_threads
    num_threads <- tryCatch({
      if (!is.null(rcpp_status$info$num_threads)) {
        rcpp_status$info$num_threads
      } else if (exists("rcpp_get_num_threads") && is.function(rcpp_get_num_threads)) {
        rcpp_get_num_threads()
      } else {
        1L
      }
    }, error = function(e) 1L)

    msg <- sprintf(
      "fastlpr: Rcpp acceleration enabled (%d OpenMP thread%s)",
      num_threads,
      if (num_threads == 1) "" else "s"
    )
  } else {
    # Rcpp not available - check if Pure R fallback is allowed
    if (.allow_pure_r()) {
      msg <- "fastlpr: WARNING - Rcpp not available, using pure R (slower). Set FASTLPR_ALLOW_PURE_R=0 to require Rcpp."
    } else {
      rcpp_detail <- ""
      if (!is.null(rcpp_status$error) && nzchar(rcpp_status$error)) {
        rcpp_detail <- paste0("\nRcpp error:\n  ", rcpp_status$error)
      }
      msg <- paste0(
        "fastlpr: WARNING - Rcpp acceleration not available.\n",
        "Rcpp is required for acceptable performance.",
        rcpp_detail,
        "\n\n",
        "Options:\n",
        "  1. Install Rcpp: install.packages('Rcpp')\n",
        "  2. Compile fastlpr: See src/README.md\n",
        "  3. Allow pure R fallback (SLOW): Sys.setenv(FASTLPR_ALLOW_PURE_R = '1')"
      )
    }
  }

  return(msg)
}

#' Package load hook
#'
#' Called automatically when the package is loaded via library() or require().
#' Reports Rcpp acceleration status to the user.
#' Disables JIT compilation to prevent 15+ second overhead on first calls.
#'
#' @param libname Library path
#' @param pkgname Package name
#' @noRd
.onLoad <- function(libname, pkgname) {
  # Disable JIT compilation to prevent 15+ second overhead
  # Root cause: R's JIT (level 3) recompiles large closures in get_lwp_estimator()
  # on the second call, causing massive delays. Disabling JIT avoids this.
  # See: https://bugs.r-project.org/show_bug.cgi?id=18555
  # Performance is actually BETTER without JIT for this package due to the
  # complex closure structures in lwp_estimator.R
  old_jit <- compiler::enableJIT(0)

  # Store old JIT level in options for restoration on unload
  options(fastlpr.old.jit.level = old_jit)
}

#' Package attach hook
#'
#' Called automatically when the package is attached via library() or require().
#' Reports Rcpp acceleration status to the user.
#'
#' @param libname Library path
#' @param pkgname Package name
#' @noRd
.onAttach <- function(libname, pkgname) {
  # Build startup message including JIT status
  msg <- .fastlpr_startup_message()
  old_jit <- getOption("fastlpr.old.jit.level", default = 3)
  jit_msg <- sprintf("fastlpr: JIT disabled (was level %d) to avoid 15s overhead", old_jit)

  # Use packageStartupMessage so it can be suppressed with
  # suppressPackageStartupMessages()
  packageStartupMessage(msg)
  packageStartupMessage(jit_msg)
}

#' Package unload hook
#'
#' Called automatically when the package is unloaded.
#' Restores the original JIT compilation level.
#'
#' @param libpath Library path
#' @noRd
.onUnload <- function(libpath) {
  # Restore original JIT level
  old_jit <- getOption("fastlpr.old.jit.level")
  if (!is.null(old_jit)) {
    compiler::enableJIT(old_jit)
    # Clean up the option
    options(fastlpr.old.jit.level = NULL)
  }
}

#' Print fastlpr startup diagnostics
#'
#' Manually prints the Rcpp acceleration status. Useful when the package
#' is loaded via source() rather than library().
#'
#' @noRd
#' @examples
#' \dontrun{
#' fastlpr_startup()
#' }
fastlpr_startup <- function() {
  msg <- .fastlpr_startup_message()
  message(msg)
  invisible(msg)
}
