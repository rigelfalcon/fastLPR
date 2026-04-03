# Copyright (c) 2024-2025 Ying Wang, Min Li
# SPDX-License-Identifier: GPL-3.0-or-later

#' Compact regression structure
#'
#' Mirrors: fastLPR/utility/core/fastLPR_compact.m
#'
#' @keywords internal
fastlpr_compact <- function(regs) {
  keep_fields <- c(
    "xraw", "yraw", "x", "y", "Tx", "Ty", "dx", "dy",
    "opt", "h", "dh", "yhat", "yhat_data", "dof",
    "xlist", "gcv_yhat", "fpp_yhat",
    "fpp_s0", "s0",
    # DOF fields (added 2025-11-21)
    "df_m", "df_sd", "pdof_m", "pdof_sd",
    "pdof_inv_m", "pdof_inv_sd"
  )
  compact <- list()
  for (field in keep_fields) {
    if (field %in% names(regs)) {
      compact[[field]] <- regs[[field]]
    }
  }
  compact
}
