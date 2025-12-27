#!/bin/bash

# activate.sh - Activate the virtual environment

VENV_NAME="llm-from-scratch-venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment '$VENV_NAME' not found."
    echo "Would you like to install it? (y/n): "
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Running installation..."
        bash "$SCRIPT_DIR/install.sh"

        # After installation, activate the environment
        source "$VENV_PATH/bin/activate"
        echo "Virtual environment activated!"
    else
        echo "Installation cancelled."
        return 1
    fi
else
    # Activate the existing virtual environment
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment '$VENV_NAME' activated!"
fi

# Show Python version and location for confirmation
echo "Python version: $(python --version)"
echo "Python bin: $(which python)"
