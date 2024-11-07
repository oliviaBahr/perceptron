#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Install Python packages from requirements.txt
echo "Installing Python packages..."
pip install -r requirements.txt

# Install pre-commit
echo "Installing pre-commit..."
pip install pre-commit

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

echo "Setup complete! Your environment is ready."
