# 🚀 Deployment Guide

This guide covers deploying the ASIA SCI Prediction web application to various platforms.

## 📋 Prerequisites

- GitHub account
- Your repository pushed to GitHub
- Model files (.pkl) included in the repository

## 🌐 Deployment Options

### Option 1: Render (Recommended - Free Tier Available)

1. **Create a Render account** at https://render.com

2. **Create a new Web Service**
   - Connect your GitHub repository
   - Name: `asia-sci-prediction`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

3. **Configure**
   - Instance Type: Free (or paid for better performance)
   - Click "Create Web Service"

4. **Access your app**
   - Render will provide a URL like: `https://asia-sci-prediction.onrender.com`

**Note**: Free tier may sleep after inactivity. First request after sleeping takes ~30 seconds.

---

### Option 2: Railway (Easy & Fast)

1. **Create a Railway account** at https://railway.app

2. **Deploy from GitHub**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway auto-detects Python and uses Procfile

3. **Configure**
   - Railway automatically sets up environment
   - Generates a public URL

4. **Access your app**
   - Railway provides a URL like: `https://asia-sci-prediction.up.railway.app`

---

### Option 3: Heroku

1. **Create a Heroku account** at https://heroku.com

2. **Install Heroku CLI**
```bash
# macOS
brew tap heroku/brew && brew install heroku

# Or download from https://devcenter.heroku.com/articles/heroku-cli
```

3. **Login to Heroku**
```bash
heroku login
```

4. **Create and deploy**
```bash
cd /path/to/asia-sci-prediction
heroku create asia-sci-prediction  # or your preferred name

# Deploy
git push heroku main

# Open in browser
heroku open
```

5. **Scale the app**
```bash
heroku ps:scale web=1
```

**Note**: Heroku no longer offers a free tier. Eco dynos start at $5/month.

---

### Option 4: PythonAnywhere (Free Tier Available)

1. **Create account** at https://www.pythonanywhere.com

2. **Upload your files**
   - Use Files tab or git clone
   - Upload all project files

3. **Create virtual environment**
```bash
mkvirtualenv --python=/usr/bin/python3.9 myenv
pip install -r requirements.txt
```

4. **Configure Web App**
   - Go to Web tab
   - Add new web app
   - Choose Flask
   - Point to your `app.py`
   - Set working directory

5. **Access your app**
   - URL: `https://yourusername.pythonanywhere.com`

---

### Option 5: Google Cloud Run (Serverless)

1. **Install Google Cloud SDK**
```bash
# Follow instructions at https://cloud.google.com/sdk/docs/install
```

2. **Create Dockerfile** (if not exists)
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
```

3. **Deploy**
```bash
# Build and deploy
gcloud run deploy asia-sci-prediction \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

### Option 6: AWS Elastic Beanstalk

1. **Install EB CLI**
```bash
pip install awsebcli
```

2. **Initialize and deploy**
```bash
eb init -p python-3.9 asia-sci-prediction
eb create asia-sci-production
eb open
```

---

### Option 7: DigitalOcean App Platform

1. **Create DigitalOcean account** at https://digitalocean.com

2. **Create App**
   - Go to Apps in dashboard
   - Connect GitHub repository
   - DigitalOcean auto-detects Python app

3. **Configure**
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `gunicorn --worker-tmp-dir /dev/shm app:app`

4. **Deploy**
   - Click "Create Resources"
   - Access provided URL

---

## 🔧 Environment Variables

If you need to set environment variables (e.g., for API keys):

```bash
# Render / Railway / Heroku
# Set in dashboard UI

# Or via CLI:
heroku config:set SECRET_KEY=your_secret_key
```

In `app.py`, access via:
```python
import os
secret_key = os.environ.get('SECRET_KEY', 'default_value')
```

---

## 📊 Performance Optimization

### For production deployment:

1. **Increase workers** in Procfile:
```
web: gunicorn --workers 4 --threads 2 app:app
```

2. **Add caching** (Redis):
```python
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'redis'})
```

3. **Use CDN** for static files

4. **Enable GZIP compression**:
```python
from flask_compress import Compress
Compress(app)
```

---

## 🔒 Security Considerations

1. **HTTPS**: Most platforms provide free HTTPS
2. **Rate limiting**: Add Flask-Limiter
3. **Input validation**: Already implemented in app.py
4. **CORS**: Add Flask-CORS if needed for API access

---

## 📈 Monitoring

### Recommended monitoring tools:

1. **Sentry** - Error tracking
```bash
pip install sentry-sdk[flask]
```

2. **New Relic** - Performance monitoring

3. **Platform-specific** - Most platforms include basic monitoring

---

## 💰 Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| Render | ✅ (sleeps after inactivity) | $7/mo | Small projects |
| Railway | ✅ $5 credit | $5/mo per GB RAM | Fast deployment |
| PythonAnywhere | ✅ Limited | $5/mo | Beginners |
| Heroku | ❌ | $5/mo (Eco) | Mature apps |
| Google Cloud Run | ✅ Generous | Pay-as-you-go | Scalability |
| DigitalOcean | ❌ | $5/mo | Full control |

---

## 🎯 Recommended Setup for This Project

**For demo/testing**: Render Free Tier
**For production**: Railway or Google Cloud Run
**For academic use**: PythonAnywhere Free Tier

---

## 🆘 Troubleshooting

### App won't start
- Check logs: Platform dashboards show deployment logs
- Verify `requirements.txt` has all dependencies
- Ensure model files (.pkl) are included

### App is slow
- Free tiers "sleep" after inactivity
- Upgrade to paid tier or use serverless (Cloud Run)
- Optimize model loading (cache in memory)

### Model file too large
- Use Git LFS for files >50MB
- Or host models separately (S3, Google Cloud Storage)
- Download models on first request

---

## 📝 Post-Deployment Checklist

- [ ] Test all prediction functionality
- [ ] Verify both models load correctly
- [ ] Test form submission and results display
- [ ] Check mobile responsiveness
- [ ] Test About page
- [ ] Set up custom domain (optional)
- [ ] Enable monitoring/logging
- [ ] Add rate limiting (optional)

---

## 🔗 Additional Resources

- [Render Docs](https://render.com/docs)
- [Railway Docs](https://docs.railway.app/)
- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [PythonAnywhere Help](https://help.pythonanywhere.com/)
- [Google Cloud Run Docs](https://cloud.google.com/run/docs)

---

**Need help?** Open an issue on GitHub or check the platform-specific documentation.

