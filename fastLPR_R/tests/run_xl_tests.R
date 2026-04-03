# Run all cross-language tests and summarize results
# Set NOT_CRAN=true to ensure no tests are skipped
Sys.setenv(NOT_CRAN = "true")
library(testthat)

files <- list.files("tests/testthat", pattern = "^test-xl-.*[.]R$", full.names = TRUE)
cat("\n=== Cross-Language Test Summary (", length(files), "tests) ===\n\n")

total_pass <- 0; total_fail <- 0; total_warn <- 0; total_skip <- 0

for (f in files) {
  res <- suppressMessages(test_file(f, reporter = "minimal"))
  df <- as.data.frame(res)
  passed <- sum(df$passed)
  failed <- sum(df$failed)
  warned <- sum(df$warning)
  skipped <- sum(df$skipped)
  status <- if (failed == 0) "PASS" else "FAIL"
  cat(sprintf("%-30s %s (P:%d F:%d W:%d S:%d)\n", basename(f), status, passed, failed, warned, skipped))
  total_pass <- total_pass + passed
  total_fail <- total_fail + failed
  total_warn <- total_warn + warned
  total_skip <- total_skip + skipped
}

cat(sprintf("\n=== TOTAL: PASS %d, FAIL %d, WARN %d, SKIP %d ===\n", total_pass, total_fail, total_warn, total_skip))
