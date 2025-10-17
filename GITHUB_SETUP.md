# 🚀 GitHub Setup & Deployment Guide

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in the details:
   - **Repository name:** `asia-sci-prediction`
   - **Description:** `Machine learning models for ASIA spinal cord injury outcome prediction with web interface`
   - **Visibility:** Public (so GitHub Pages works)
   - **DO NOT** check "Initialize with README" (we already have one)
3. Click **"Create repository"**

## Step 2: Push Your Code to GitHub

After creating the repository, run these commands in Terminal:

```bash
cd /Users/aaryanpatel/Desktop/ASIA_motor_ml_training

# Add GitHub as remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/asia-sci-prediction.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note:** You may be prompted to authenticate. Use your GitHub Personal Access Token (not password).

### If you need a Personal Access Token:
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "ASIA SCI Project"
4. Check the "repo" scope
5. Click "Generate token"
6. Copy the token and use it as your password when prompted

## Step 3: Deploy Web App to Render (Free)

The Flask app needs a server. GitHub Pages only hosts static files, so use Render:

1. Go to https://render.com
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your repository: `asia-sci-prediction`
5. Configure:
   - **Name:** `asia-sci-prediction`
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
6. Click "Create Web Service"
7. Wait ~5 minutes for deployment

You'll get a URL like: `https://asia-sci-prediction.onrender.com`

## Step 4: Setup GitHub Pages for Data Dictionary

The data dictionary viewer can go on GitHub Pages:

1. Go to your repository on GitHub
2. Click "Settings" → "Pages"
3. Under "Source", select "Deploy from a branch"
4. Select branch: `main`
5. Select folder: `/github_pages_site`
6. Click "Save"
7. Wait 2-3 minutes

Your data dictionary will be at:
`https://YOUR_USERNAME.github.io/asia-sci-prediction/`

## Alternative: Deploy Everything to Railway (Easier)

Railway is faster and auto-deploys:

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `asia-sci-prediction`
5. Railway auto-detects Python and uses your Procfile
6. You get a URL immediately!

## Summary

**Main Web App:** Use Render or Railway (free hosting)
**Data Dictionary:** Use GitHub Pages (static hosting)
**Code Repository:** GitHub (version control)

## Quick Start Commands

```bash
# Set your GitHub username here
GITHUB_USERNAME="your-username-here"

# Add remote and push
cd /Users/aaryanpatel/Desktop/ASIA_motor_ml_training
git remote add origin https://github.com/$GITHUB_USERNAME/asia-sci-prediction.git
git push -u origin main
```

## Need Help?

If you get authentication errors:
- Use Personal Access Token instead of password
- Or use SSH: `git remote set-url origin git@github.com:YOUR_USERNAME/asia-sci-prediction.git`

If push fails:
- Make sure you created the repo first on GitHub
- Check your username is correct in the URL

