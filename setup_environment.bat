@echo off
title EV Charging Station AI - One-Click Environment Setup
color 0A

echo ======================================================================
echo   ⚡ EV Charging Station AI Assistant - Automated Environment Setup
echo ======================================================================
echo.

:: 1. Check Python installation
echo [*] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

python --version
echo.

:: 2. Upgrade pip
echo [*] Upgrading pip package manager...
python -m pip install --upgrade pip
echo.

:: 3. Install PyTorch & Torchvision (CPU version for universal compatibility)
echo [*] Installing PyTorch and Torchvision (CPU build)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.

:: 4. Install remaining required dependencies
echo [*] Installing core project dependencies from requirements.txt...
python -m pip install -r requirements.txt
echo.

:: 5. Run verification script
echo [*] Verifying installation and model checkpoints...
python verify_setup.py

echo.
echo ======================================================================
echo   Setup Complete! To launch the dashboard, run:
echo     streamlit run app_member3.py
echo ======================================================================
echo.
pause
