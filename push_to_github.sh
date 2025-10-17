#!/bin/bash

# ASIA SCI Prediction - Push to GitHub Script
# Run this AFTER creating the repository on GitHub

echo "=========================================="
echo "Pushing ASIA SCI Prediction to GitHub"
echo "=========================================="
echo ""

cd /Users/aaryanpatel/Desktop/ASIA_motor_ml_training

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo "✓ Remote 'origin' already exists"
    git remote -v
else
    echo "Adding GitHub remote..."
    git remote add origin https://github.com/superpatel101/asia-sci-prediction.git
    echo "✓ Remote added"
fi

echo ""
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "=========================================="
echo "✓ Code pushed to GitHub!"
echo "=========================================="
echo ""
echo "Your repository is now at:"
echo "https://github.com/superpatel101/asia-sci-prediction"
echo ""
echo "Next steps:"
echo "1. Visit your repo on GitHub"
echo "2. Go to Settings → Pages"
echo "3. Deploy the data dictionary from /github_pages_site"
echo "4. Deploy the web app to Render.com (see GITHUB_SETUP.md)"
echo ""

