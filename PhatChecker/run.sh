#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Define the path to the python executable within the venv
# Note: On Windows/Git Bash, this might be .venv/Scripts/python
VENV_PYTHON=".venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment not found or corrupted. Cleaning and setting up..."
    
    # 1. Clear out any broken attempts
    rm -rf .venv
    
    # 2. Re-create the venv
    python3 -m venv .venv || { echo "Error: Failed to create venv. Is 'python3-venv' installed?"; exit 1; }
    
    # 3. Double-check the file exists now
    if [ ! -f "$VENV_PYTHON" ]; then
        echo "Error: .venv/bin/python still not found after creation."
        exit 1
    fi
    
    echo "Installing dependencies..."
    $VENV_PYTHON -m pip install --upgrade pip -q
    if [ -f "requirements.txt" ]; then
        $VENV_PYTHON -m pip install -r requirements.txt -q
    fi
    echo "Setup complete!"
else
    echo "Virtual environment found. Starting the app..."
fi

# Run the application using the variable we defined
$VENV_PYTHON visual_diff_app.py