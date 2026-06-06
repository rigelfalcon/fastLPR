"""
Master Script to Reproduce All Figures from the fastLPR Paper

This script runs all individual figure generation scripts and creates a summary report.
It ensures reproducibility by running all examples in sequence and collecting timing
and status information.

Usage:
    python reproduce_all_figures.py

The script will:
  1. Run all 5 figure generation scripts (Fig 2-6)
  2. Collect timing and status information
  3. Generate a summary report
  4. Save all figures to ../fig/reproduced/

Copyright (c) 2024 fastLPR Development Team
"""

import subprocess
import time
import os
import sys
from datetime import datetime

print()
print("=" * 80)
print("Reproducing All Figures from the fastLPR Paper")
print("=" * 80)
print()

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define all figure scripts
figure_scripts = [
    {
        "name": "Figure 2: Kernel Density Estimation (1D, 2D, 3D)",
        "script": "example_kde.py",
        "description": "KDE with bandwidth selection in 1D, 2D, and 3D",
    },
    {
        "name": "Figure 3: Boundary Comparison (NW vs LL vs LQ)",
        "script": "example_boundary.py",
        "description": "Comparison of local polynomial regression orders",
    },
    {
        "name": "Figure 4: Complex-Valued Regression",
        "script": "example_complex.py",
        "description": "Regression with complex-valued responses (log(z))",
    },
    {
        "name": "Figure 5: Heteroscedastic Regression",
        "script": "example_hetero.py",
        "description": "Mean and variance estimation in 1D and 2D",
    },
    {
        "name": "Figure 6: Real-World Applications",
        "script": "example_qeeg.py",
        "description": "qEEG log-spectrum and MRI T1 brain image regression",
    },
]

# Results storage
results = []

# Run each script
total_start_time = time.time()

for i, fig_info in enumerate(figure_scripts, 1):
    print(f"\n{'=' * 80}")
    print(f"Running {i}/{len(figure_scripts)}: {fig_info['name']}")
    print(f"{'=' * 80}")
    print(f"Description: {fig_info['description']}")
    print(f"Script: {fig_info['script']}")
    print()

    script_path = os.path.join(script_dir, fig_info["script"])

    # Check if script exists
    if not os.path.isfile(script_path):
        print(f"ERROR: Script not found: {script_path}")
        results.append(
            {
                "name": fig_info["name"],
                "script": fig_info["script"],
                "status": "FAILED",
                "error": "Script not found",
                "time": 0,
            }
        )
        continue

    # Run the script
    start_time = time.time()
    try:
        # Run with subprocess, capturing output
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[SUCCESS] Completed in {elapsed_time:.2f} seconds")
            results.append(
                {
                    "name": fig_info["name"],
                    "script": fig_info["script"],
                    "status": "SUCCESS",
                    "error": None,
                    "time": elapsed_time,
                }
            )
        else:
            print(f"\n[FAILED] Return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            results.append(
                {
                    "name": fig_info["name"],
                    "script": fig_info["script"],
                    "status": "FAILED",
                    "error": f"Return code {result.returncode}",
                    "time": elapsed_time,
                }
            )

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"\n[FAILED] Timeout after {elapsed_time:.2f} seconds")
        results.append(
            {
                "name": fig_info["name"],
                "script": fig_info["script"],
                "status": "TIMEOUT",
                "error": "Execution timeout (30 minutes)",
                "time": elapsed_time,
            }
        )

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n[FAILED] {str(e)}")
        results.append(
            {
                "name": fig_info["name"],
                "script": fig_info["script"],
                "status": "ERROR",
                "error": str(e),
                "time": elapsed_time,
            }
        )

total_elapsed_time = time.time() - total_start_time

################################################################################
# Generate Summary Report
################################################################################

print()
print("=" * 80)
print("SUMMARY REPORT")
print("=" * 80)
print()

# Count successes and failures
n_success = sum(1 for r in results if r["status"] == "SUCCESS")
n_failed = len(results) - n_success

print(f"Total figures: {len(results)}")
print(f"Successful: {n_success}")
print(f"Failed: {n_failed}")
print(
    f"Total time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)"
)
print()

# Detailed results
print("Detailed Results:")
print("-" * 80)
for i, result in enumerate(results, 1):
    status_symbol = "[OK]" if result["status"] == "SUCCESS" else "[FAIL]"
    print(f"{i}. {status_symbol} {result['name']}")
    print(f"   Script: {result['script']}")
    print(f"   Status: {result['status']}")
    if result["error"]:
        print(f"   Error: {result['error']}")
    print(f"   Time: {result['time']:.2f} seconds")
    print()

# Check output directory
fig_dir = os.path.join(script_dir, "..", "fig", "reproduced")
if os.path.isdir(fig_dir):
    png_files = [f for f in os.listdir(fig_dir) if f.endswith("_python.png")]
    pdf_files = [f for f in os.listdir(fig_dir) if f.endswith("_python.pdf")]

    print(f"Output Directory: {os.path.abspath(fig_dir)}")
    print(f"PNG files generated: {len(png_files)}")
    print(f"PDF files generated: {len(pdf_files)}")
    print()

    if png_files:
        print("Generated PNG files:")
        for f in sorted(png_files):
            print(f"  - {f}")
        print()

    if pdf_files:
        print("Generated PDF files:")
        for f in sorted(pdf_files):
            print(f"  - {f}")
        print()

################################################################################
# Save Summary Report to File
################################################################################

report_path = os.path.join(fig_dir, "reproduction_report.txt")
try:
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("fastLPR Paper Figure Reproduction Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write("\n")

        f.write(f"Total figures: {len(results)}\n")
        f.write(f"Successful: {n_success}\n")
        f.write(f"Failed: {n_failed}\n")
        f.write(
            f"Total time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)\n"
        )
        f.write("\n")

        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        for i, result in enumerate(results, 1):
            status_symbol = "[OK]" if result["status"] == "SUCCESS" else "[FAIL]"
            f.write(f"{i}. {status_symbol} {result['name']}\n")
            f.write(f"   Script: {result['script']}\n")
            f.write(f"   Status: {result['status']}\n")
            if result["error"]:
                f.write(f"   Error: {result['error']}\n")
            f.write(f"   Time: {result['time']:.2f} seconds\n")
            f.write("\n")

        f.write(f"\nOutput Directory: {os.path.abspath(fig_dir)}\n")
        f.write(f"PNG files generated: {len(png_files)}\n")
        f.write(f"PDF files generated: {len(pdf_files)}\n")

    print(f"Summary report saved to: {os.path.abspath(report_path)}")
    print()

except Exception as e:
    print(f"Warning: Could not save summary report: {e}")
    print()

################################################################################
# Final Status
################################################################################

if n_failed == 0:
    print("=" * 80)
    print("[SUCCESS] ALL FIGURES REPRODUCED SUCCESSFULLY!")
    print("=" * 80)
    sys.exit(0)
else:
    print("=" * 80)
    print(f"[FAILED] {n_failed} FIGURE(S) FAILED TO REPRODUCE")
    print("=" * 80)
    sys.exit(1)
