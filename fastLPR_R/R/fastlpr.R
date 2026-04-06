# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

# fastlpr.R — Legacy monolithic implementation (DEPRECATED)
#
# This file previously contained the original single-file fastLPR implementation.
# All functionality has been moved to modular files:
#   - cv_fastlpr.R: main API function cv_fastlpr()
#   - cv_fastkde.R: main API function cv_fastkde()
#   - plotting.R:   S3 plot methods for fastlpr_result and fastkde_result
#   - design_matrix.R, nufft.R, etc.: internal implementation details
#
# The dead code was removed to prevent confusion and R CMD check warnings
# about undefined functions (fastlpr_compute_kdf, fastlpr_inufft, fastkde_eval).
