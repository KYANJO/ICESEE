@echo off
:: setup_venv.bat

:: Create virtual environment
python -m venv venv

:: Get project directory
set "SCRIPT_DIR=%~dp0"

:: Add project/ to sitecustomize.py
for /f "delims=" %%i in ('dir venv\lib\site-packages /a:d /b') do set "SITE_PACKAGES=venv\lib\site-packages\%%i"
echo import sys > "%SITE_PACKAGES%\sitecustomize.py"
echo sys.path.append('%SCRIPT_DIR%') >> "%SITE_PACKAGES%\sitecustomize.py"

echo Virtual environment created with PYTHONPATH including %SCRIPT_DIR%
echo Activate with: venv\Scripts\activate
echo Then, run 'make install' to install ICESEE (recommended) or use PYTHONPATH.
