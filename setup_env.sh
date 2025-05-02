#!/bin/bash
# setup_venv.sh

# Create virtual environment
python -m venv venv

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add project/ to sitecustomize.py
SITE_PACKAGES=$(find venv/lib -type d -name "site-packages" | head -n 1)
mkdir -p "$SITE_PACKAGES"
echo "import sys" > "$SITE_PACKAGES/sitecustomize.py"
echo "sys.path.append('$SCRIPT_DIR')" >> "$SITE_PACKAGES/sitecustomize.py"

echo "Virtual environment created with PYTHONPATH including $SCRIPT_DIR"
echo "Activate with: source venv/bin/activate"
echo "Then, run 'make install' to install ICESEE (recommended) or use PYTHONPATH."
