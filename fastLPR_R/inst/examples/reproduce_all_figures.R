#!/usr/bin/env Rscript
# Run all R examples and time them

cat("================================================================================\n")
cat("Running All fastLPR R Examples\n")
cat("================================================================================\n\n")

# Change to project root (auto-detect from script location)
script_dir <- tryCatch({
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  "."
})
# Navigate from examples/ to repo root
repo_root <- normalizePath(file.path(script_dir, "../.."), mustWork = FALSE)
if (dir.exists(file.path(repo_root, "fastLPR_R"))) {
  setwd(repo_root)
} else {
  # Fallback: assume already in repo root
  setwd(getwd())
}

# Track results
results <- list()
total_time <- 0

# Function to run an example and time it
run_example <- function(example_name, example_file) {
  cat(sprintf("\n>>> Running: %s <<<\n", example_name))
  cat(sprintf("File: %s\n", example_file))
  cat("----------------------------------------\n")

  start_time <- Sys.time()

  result <- tryCatch({
    source(example_file, local = TRUE)
    list(status = "SUCCESS", error = NULL)
  }, error = function(e) {
    list(status = "FAILED", error = e$message)
  })

  end_time <- Sys.time()
  elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))

  cat(sprintf("\nStatus: %s\n", result$status))
  cat(sprintf("Time: %.2f seconds\n", elapsed))

  if (result$status == "FAILED") {
    cat(sprintf("Error: %s\n", result$error))
  }

  return(list(
    name = example_name,
    file = example_file,
    status = result$status,
    time = elapsed,
    error = result$error
  ))
}

# Run all examples
results[[1]] <- run_example(
  "Example 1: 1D Kernel Density Estimation",
  "fastLPR_R/examples/example_kde_1d.R"
)

results[[2]] <- run_example(
  "Example 2: 1D Regression (Orders 0 and 1)",
  "fastLPR_R/examples/example_regression_1d.R"
)

results[[3]] <- run_example(
  "Figure 2: FastKDE (1D and 2D)",
  "fastLPR_R/examples/example_kde.R"
)

results[[4]] <- run_example(
  "Figure 3: Boundary Comparison (Orders 0, 1, 2)",
  "fastLPR_R/examples/example_boundary.R"
)

results[[5]] <- run_example(
  "Figure 4: Complex-Valued Regression",
  "fastLPR_R/examples/example_complex.R"
)

results[[6]] <- run_example(
  "Figure 5: Heteroscedasticity (1D and 2D)",
  "fastLPR_R/examples/example_hetero.R"
)

results[[7]] <- run_example(
  "Figure 6: Real Applications (qEEG and MRI)",
  "fastLPR_R/examples/example_qeeg.R"
)

# Summary
cat("\n================================================================================\n")
cat("SUMMARY\n")
cat("================================================================================\n\n")

total_time <- sum(sapply(results, function(r) r$time))
successes <- sum(sapply(results, function(r) r$status == "SUCCESS"))
failures <- sum(sapply(results, function(r) r$status == "FAILED"))

for (i in seq_along(results)) {
  r <- results[[i]]
  status_symbol <- if (r$status == "SUCCESS") "✅" else "❌"
  cat(sprintf("%s Example %d: %s\n", status_symbol, i, r$name))
  cat(sprintf("   Time: %.2f seconds\n", r$time))
  if (r$status == "FAILED") {
    cat(sprintf("   Error: %s\n", r$error))
  }
}

cat(sprintf("\nTotal Examples: %d\n", length(results)))
cat(sprintf("Successes: %d\n", successes))
cat(sprintf("Failures: %d\n", failures))
cat(sprintf("Total Time: %.2f seconds\n", total_time))

if (failures == 0) {
  cat("\n🎉 ALL EXAMPLES PASSED!\n")
} else {
  cat(sprintf("\n⚠️  %d EXAMPLE(S) FAILED\n", failures))
}

cat("\n================================================================================\n")
