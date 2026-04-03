@echo off
REM ################################################################################
REM Reproduce All Figures from fastLPR JSS Paper (Windows Version)
REM
REM This script automates the reproduction of all figures (Fig 2-6) from the
REM Journal of Statistical Software (JSS) submission for the fastLPR package.
REM
REM Usage:
REM   reproduce_paper.bat
REM
REM Prerequisites:
REM   - Python 3.9+ installed and in PATH
REM   - fastLPR package installed (pip install -e .)
REM   - Dependencies installed (pip install -r requirements.txt)
REM
REM Output:
REM   - Figures saved to fastLPR_py\fig\reproduced\
REM   - Summary report printed to console
REM   - Log file saved to replication\reproduction_log.txt
REM
REM Copyright (c) 2024 Ying Wang, Min Li
REM License: GPL-3.0-or-later
REM ################################################################################

setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo Reproducing All Figures from fastLPR JSS Paper
echo ================================================================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PYTHON_PKG_DIR=%PROJECT_ROOT%\fastLPR_py

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add to PATH
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo Python version: %PYTHON_VERSION%

REM Check if fastLPR package directory exists
if not exist "%PYTHON_PKG_DIR%" (
    echo [ERROR] fastLPR_py directory not found at %PYTHON_PKG_DIR%
    exit /b 1
)

REM Check if fastLPR is installed
python -c "import fastlpr" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] fastLPR package not installed
    echo Installing fastLPR package...
    cd /d "%PYTHON_PKG_DIR%"
    pip install -e .
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to install fastLPR package
        exit /b 1
    )
    echo [SUCCESS] fastLPR package installed successfully
)

REM Check if dependencies are installed
echo.
echo Checking dependencies...
python -c "import numpy; import scipy; import matplotlib" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Missing dependencies
    echo Installing dependencies from requirements.txt...
    pip install -r "%SCRIPT_DIR%requirements.txt"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to install dependencies
        exit /b 1
    )
    echo [SUCCESS] Dependencies installed successfully
)

REM Print dependency versions
echo.
echo Dependency versions:
python -c "import numpy; print('  numpy:', numpy.__version__)"
python -c "import scipy; print('  scipy:', scipy.__version__)"
python -c "import matplotlib; print('  matplotlib:', matplotlib.__version__)"

REM Create output directory
set OUTPUT_DIR=%PYTHON_PKG_DIR%\fig\reproduced
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
echo.
echo Output directory: %OUTPUT_DIR%

REM Run the master reproduction script
echo.
echo ================================================================================
echo Running Reproduction Script
echo ================================================================================
echo.

cd /d "%PYTHON_PKG_DIR%\examples"
set LOG_FILE=%SCRIPT_DIR%reproduction_log.txt

REM Run with timestamp
set START_TIME=%TIME%
python reproduce_all_figures.py 2>&1 | tee "%LOG_FILE%"
set EXIT_CODE=%ERRORLEVEL%
set END_TIME=%TIME%

echo.
echo ================================================================================
echo Reproduction Complete
echo ================================================================================
echo.

if %EXIT_CODE% equ 0 (
    echo [SUCCESS] All figures reproduced successfully
    echo.
    echo Output directory: %OUTPUT_DIR%
    echo Log file: %LOG_FILE%
    echo.

    REM Count generated files
    set PNG_COUNT=0
    set PDF_COUNT=0
    for %%F in ("%OUTPUT_DIR%\*_python.png") do set /a PNG_COUNT+=1
    for %%F in ("%OUTPUT_DIR%\*_python.pdf") do set /a PDF_COUNT+=1

    echo Generated files:
    echo   PNG files: !PNG_COUNT!
    echo   PDF files: !PDF_COUNT!
    echo.

    REM List generated PNG files
    if !PNG_COUNT! gtr 0 (
        echo PNG files:
        for %%F in ("%OUTPUT_DIR%\*_python.png") do echo   - %%~nxF
        echo.
    )

    exit /b 0
) else (
    echo [FAILED] Some figures failed to reproduce
    echo.
    echo Log file: %LOG_FILE%
    echo.
    echo Please check the log file for details on failures.
    echo.
    exit /b 1
)
