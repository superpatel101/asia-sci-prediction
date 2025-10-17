# 🚀 Quick Start Guide

Get the ASIA SCI Prediction web app running in 3 minutes!

## ⚡ Super Quick Start (Local)

```bash
# 1. Navigate to project directory
cd /Users/aaryanpatel/Desktop/ASIA_motor_ml_training

# 2. Install dependencies (if not already installed)
pip3 install flask gunicorn

# 3. Run the app
python3 app.py
```

Open your browser to: **http://localhost:5000**

That's it! 🎉

---

## 📦 Proper Setup (Recommended)

### Step 1: Create Virtual Environment

```bash
cd /Users/aaryanpatel/Desktop/ASIA_motor_ml_training

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows
```

### Step 2: Install All Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the App

```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 🌐 Deploy to the Internet

### Option A: Render (Free, Easy)

1. Go to https://render.com
2. Sign up / Log in
3. Click "New +" → "Web Service"
4. Connect your GitHub repo
5. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
6. Click "Create Web Service"
7. Wait ~5 minutes for deployment
8. Your app is live! 🌐

### Option B: Railway (Fast)

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-configures everything
6. Your app is live in 3 minutes! ⚡

See [DEPLOYMENT.md](DEPLOYMENT.md) for more options.

---

## 📊 Using the Web Interface

### Basic Workflow:

1. **Enter Patient Data**
   - Fill in ASIA scores (most important)
   - Add demographics and injury details
   - Optional: Expand socioeconomic section

2. **Click "Predict Discharge Outcomes"**

3. **View Results**
   - Motor Score Prediction (left)
   - Impairment Grade Prediction (right)
   - Confidence levels and probabilities

### Required Fields:

- ⭐ ASIA Motor Score at Admission (0-100)
- ⭐ ASIA Impairment Grade at Admission (A/B/C/D)
- ⭐ Neurological Level (e.g., C5, T4, L1)
- ⭐ Age at Injury
- ⭐ Days from Injury to Rehab

---

## 🔧 Using the API

### Python Example:

```python
import requests

# Patient data
patient = {
    'admission_motor': 42,
    'admission_grade': 'C',
    'neuro_level': 'T4',
    'age': 45,
    'sex': 1,
    'days_to_rehab': 25,
    # ... other fields with defaults
}

# Make prediction
response = requests.post('http://localhost:5000/predict', data=patient)
result = response.json()

print(f"Predicted motor score: {result['motor_prediction']['predicted_score']}")
print(f"Predicted grade: {result['grade_prediction']['predicted_grade']}")
```

### JavaScript Example:

```javascript
fetch('/predict', {
    method: 'POST',
    body: new FormData(document.getElementById('predictionForm'))
})
.then(response => response.json())
.then(data => {
    console.log('Motor score:', data.motor_prediction.predicted_score);
    console.log('Grade:', data.grade_prediction.predicted_grade);
});
```

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Change port in app.py or:
python app.py  # It will auto-select a different port
```

### Models not loading
- Ensure `.pkl` files are in the project directory
- Check file permissions
- Verify scikit-learn is installed

### App is slow on first request
- Normal! Models load on first request (~2 seconds)
- Subsequent requests are fast

---

## 📁 Project Structure

```
ASIA_motor_ml_training/
├── app.py                    ← Main web application
├── predict_motor_score.py    ← Motor score predictor
├── predict_impairment_grade.py ← Grade classifier
├── templates/                ← HTML files
│   ├── index.html
│   └── about.html
├── static/                   ← CSS and JS
│   ├── css/style.css
│   └── js/script.js
├── *.pkl                     ← Trained models
└── requirements.txt          ← Dependencies
```

---

## 🎯 What You Can Do

✅ Predict motor scores at discharge  
✅ Predict impairment grades at discharge  
✅ View confidence levels and probabilities  
✅ Get clinical interpretations  
✅ Use via web interface or API  
✅ Deploy to the cloud (free!)  
✅ Integrate into other systems  

---

## 🔗 Next Steps

1. ✅ **Run locally** (you're here!)
2. 📤 **Push to GitHub** (run `./setup_github.sh`)
3. 🌐 **Deploy online** (see [DEPLOYMENT.md](DEPLOYMENT.md))
4. 📚 **Read documentation** (see [README.md](README.md))
5. 🔬 **Explore analysis** (see research figures and reports)

---

## 📧 Need Help?

- Check [README.md](README.md) for detailed info
- See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for API details
- Read [DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment
- Open an issue on GitHub

---

**Made with ❤️ for spinal cord injury research**

