#!/bin/bash
# setup_venv.sh

# Create virtual environment
python -m venv icesee-env

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add project/ to sitecustomize.py
SITE_PACKAGES=$(find icesee-env/lib -type d -name "site-packages" | head -n 1)
mkdir -p "$SITE_PACKAGES"
echo "import sys" > "$SITE_PACKAGES/sitecustomize.py"
echo "sys.path.append('$SCRIPT_DIR')" >> "$SITE_PACKAGES/sitecustomize.py"

# Install required dependencies from requirements.txt
source icesee-env/bin/activate
pip install -r requirements.txt
deactivate

echo "Virtual environment 'icesee-env' created with PYTHONPATH including $SCRIPT_DIR"
echo "Dependencies from requirements.txt installed"
echo "Activate with: source icesee-env/bin/activate"
echo "Then, run 'make install' to install ICESEE (recommended) or use PYTHONPATH."