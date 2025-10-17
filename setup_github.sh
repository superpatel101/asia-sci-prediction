#!/bin/bash

# ASIA SCI Prediction - GitHub Repository Setup Script
# This script helps you set up the GitHub repository

echo "=========================================="
echo "ASIA SCI Prediction - GitHub Setup"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

echo "✓ Git is installed"
echo ""

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already initialized"
fi

echo ""
echo "Adding files to Git..."
git add .

echo ""
echo "Creating initial commit..."
git commit -m "Initial commit: ASIA SCI Outcome Prediction Models

- Added Random Forest models for motor score and impairment grade prediction
- Included web interface (Flask app)
- Added training scripts and analysis tools
- Included comprehensive documentation and research figures
- Model performance: R²=0.38 (motor), Accuracy=75.4% (grade)
"

echo ""
echo "=========================================="
echo "✓ Local Git repository is ready!"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name: asia-sci-prediction (or your choice)"
echo "   - Description: ML-based ASIA SCI outcome prediction with web interface"
echo "   - Keep it Public or Private (your choice)"
echo "   - Do NOT initialize with README, .gitignore, or license (we have them)"
echo ""
echo "2. Copy the repository URL from GitHub"
echo ""
echo "3. Run these commands (replace YOUR_USERNAME with your GitHub username):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/asia-sci-prediction.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "=========================================="
echo ""
echo "OPTIONAL: Set up Git LFS for large model files (.pkl)"
echo ""
echo "If your model files are large (>50MB), use Git LFS:"
echo "   git lfs install"
echo "   git lfs track \"*.pkl\""
echo "   git add .gitattributes"
echo "   git commit -m \"Add Git LFS tracking for model files\""
echo ""
echo "=========================================="
echo ""
echo "After pushing to GitHub, you can:"
echo "  - Enable GitHub Pages for the data dictionary viewer"
echo "  - Deploy the web app to Heroku, Render, or Railway"
echo "  - Set up GitHub Actions for CI/CD"
echo ""

