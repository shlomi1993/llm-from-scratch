#!/bin/bash

# install.sh - Create virtual environment and install requirements

set -e  # Exit on any error

VENV_NAME="llm-from-scratch-venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating virtual environment: $VENV_NAME"

# Create virtual environment with Python 3.10
python3.10 -m venv "$VENV_NAME"

echo "Virtual environment created successfully."

# Activate virtual environment
source "$VENV_NAME/bin/activate"

echo "Installing requirements..."

# Install requirements
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Installation completed successfully!"
echo "Virtual environment '$VENV_NAME' is ready to use."
echo "To activate it manually, run: source $VENV_NAME/bin/activate"
echo "Or use the activate.sh script: source activate.sh"
