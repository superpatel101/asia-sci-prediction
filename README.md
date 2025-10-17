# 🏥 ASIA Spinal Cord Injury Outcome Prediction

Machine learning-based prediction models for spinal cord injury outcomes at discharge using admission data. Features a web interface for easy clinical use.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This project provides two Random Forest models trained on the National Spinal Cord Injury Statistical Center (NSCISC) database:

1. **Motor Score Predictor**: Predicts ASIA motor score at discharge (0-100 scale)
2. **Impairment Grade Predictor**: Predicts ASIA Impairment Scale grade at discharge (A/B/C/D/E)

### Key Features

- 🎯 **High Accuracy**: 75.4% accuracy for grade classification, R²=0.38 for motor score prediction
- 🚀 **Web Interface**: User-friendly web application for easy clinical use
- 📊 **Data Visualization**: Comprehensive analysis with SHAP plots, ROC curves, and performance metrics
- 💾 **Complete Research Package**: Includes all training scripts, analysis tools, and research figures
- 🔒 **No Data Leakage**: Models use only admission-time features for truly predictive outcomes

## 🎯 Model Performance

### Motor Score Prediction Model
- **R² Score**: 0.3832 (explains 38.3% of variance)
- **RMSE**: 21.0 points
- **MAE**: 14.8 points
- **Training Data**: 10,543 patients
- **Algorithm**: Random Forest Regressor (200 trees)

### Impairment Grade Prediction Model
- **Accuracy**: 75.4%
- **F1-Score (Weighted)**: 0.7433
- **AUC (Weighted)**: 0.9178
- **Training Data**: 15,053 patients
- **Algorithm**: Random Forest Classifier (200 trees)

#### Per-Grade Performance:
| Grade | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| A (Complete) | 0.73 | 0.76 | 0.71 |
| B (Sensory Incomplete) | 0.56 | 0.59 | 0.53 |
| C (Motor <50%) | 0.76 | 0.77 | 0.75 |
| D (Motor ≥50%) | 0.68 | 0.71 | 0.66 |
| E (Normal) | 0.84 | 0.81 | 0.88 |

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/asia-sci-prediction.git
cd asia-sci-prediction
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Web Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

### Using the Prediction Models Programmatically

```python
from predict_motor_score import MotorScorePredictor
from predict_impairment_grade import ImpairmentGradePredictor

# Initialize predictors
motor_predictor = MotorScorePredictor()
grade_predictor = ImpairmentGradePredictor()

# Patient data at admission
patient_data = {
    'AInjAge': 45,              # Age
    'ASex': 1,                  # Male
    'AASATotA': 42,             # Motor score at admission
    'AASAImAd': 'C',            # Grade C at admission
    'ANurLvlA': 'T4',           # Neurological level
    'AI2RhADa': 25,             # Days to rehab
    # ... other features
}

# Get predictions
motor_result = motor_predictor.predict(patient_data)
grade_result = grade_predictor.predict(patient_data)

print(f"Predicted discharge motor score: {motor_result['predicted_discharge_motor_score']:.1f}")
print(f"Predicted discharge grade: {grade_result['predicted_discharge_grade']}")
```

## 📂 Project Structure

```
asia-sci-prediction/
├── app.py                              # Flask web application
├── predict_motor_score.py              # Motor score inference script
├── predict_impairment_grade.py         # Grade classification inference script
├── train_motor_clean_model.py          # Motor score model training
├── train_impairment_classifier.py      # Grade classification model training
├── generate_clean_motor_shap.py        # SHAP analysis for motor model
├── generate_shap_and_roc.py            # SHAP and ROC for grade model
├── analyze_motor_improvement_by_grade.py # Motor recovery analysis
├── templates/                          # HTML templates
│   ├── index.html                      # Main prediction interface
│   └── about.html                      # About page
├── static/                             # Static files
│   ├── css/style.css                   # Stylesheet
│   └── js/script.js                    # JavaScript
├── *.pkl                               # Trained models and artifacts
├── *.png                               # Generated figures
├── *.csv                               # Feature importance and results
├── requirements.txt                    # Python dependencies
├── INFERENCE_GUIDE.md                  # Detailed inference guide
└── README.md                           # This file
```

## 📊 Key Findings

### Recovery Patterns by Admission Grade
- **Grade A (Complete)**: +3.8 ± 9.3 points, 43% improve
- **Grade B (Sensory Incomplete)**: +14.9 ± 18.8 points, 79% improve
- **Grade C (Motor <50%)**: +25.6 ± 20.1 points, 91% improve ⭐ **Best recovery**
- **Grade D (Motor ≥50%)**: +12.3 ± 12.9 points, 83% improve

### Most Important Features

#### For Motor Score Prediction:
1. ASIA Motor Score at Admission (42.9%)
2. ASIA Impairment Grade at Admission (19.7%)
3. Functional Independence Score (8.5%)
4. Neurological Level (6.7%)
5. Mechanical Ventilation (5.4%)

#### For Impairment Grade Prediction:
1. ASIA Impairment Grade at Admission (38.3%)
2. Days from Injury to Rehab (14.2%)
3. Mechanical Ventilation (7.5%)
4. Neurological Level (7.4%)
5. Age at Injury (7.2%)

## 🔬 Research & Analysis

The project includes comprehensive analysis scripts and generated figures:

- **SHAP Analysis**: Feature importance and impact visualization
- **ROC Curves**: Multi-class ROC analysis with AUC metrics
- **Motor Recovery Analysis**: Detailed breakdown by admission grade
- **Statistical Reports**: Complete performance metrics and insights
- **Research PDF**: Professional compilation of all figures and results

See `INFERENCE_GUIDE.md` for detailed documentation.

## 📈 Web Interface Features

- ✅ Intuitive form for patient data entry
- ✅ Real-time predictions for both models
- ✅ Confidence levels and probability distributions
- ✅ Clinical interpretations and recommendations
- ✅ Mobile-responsive design
- ✅ Auto-save functionality (localStorage)
- ✅ Comprehensive about page with model details

## ⚠️ Limitations & Clinical Use

### Limitations
- Motor score model explains 38% of variance (62% unexplained)
- Individual outcomes can vary significantly
- Models trained on North American population
- Grade B classification has lower accuracy (F1=0.56)

### Intended Use
These models are designed to assist with:
- Early patient and family counseling
- Setting realistic expectations and goals
- Rehabilitation planning and resource allocation
- Research and quality improvement initiatives

**Important**: Predictions should complement, not replace, clinical judgment. They provide statistical likelihoods based on historical data.

## 🛠️ Development & Training

### Training New Models

```bash
# Train motor score model
python train_motor_clean_model.py

# Train impairment grade classifier
python train_impairment_classifier.py

# Generate SHAP plots and ROC curves
python generate_clean_motor_shap.py
python generate_shap_and_roc.py
```

### Analysis Scripts

```bash
# Analyze motor recovery patterns
python analyze_motor_improvement_by_grade.py

# Analyze specific features
python analyze_AI2RhADa_simple.py
```

## 📦 Deployment

### Local Development
```bash
python app.py
```

### Production (with Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Docker (optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

## 💾 Data Source

Models were trained on the National Spinal Cord Injury Statistical Center (NSCISC) database, which contains comprehensive data on spinal cord injury patients from specialized centers across North America.

- Website: https://www.nscisc.uab.edu/
- Database: De-identified patient records
- Time Period: Multi-year comprehensive dataset

## 📜 Citation

If you use this work in your research, please cite:

```
ASIA Spinal Cord Injury Outcome Prediction Models
Trained on the National Spinal Cord Injury Statistical Center (NSCISC) Database
Models: Random Forest Regressor (Motor Score) and Random Forest Classifier (Impairment Grade)
Date: October 2024
GitHub: https://github.com/yourusername/asia-sci-prediction
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Follow PEP 8 style guide
2. Add unit tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting PR

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact the maintainer.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- National Spinal Cord Injury Statistical Center (NSCISC) for the comprehensive database
- scikit-learn team for the excellent machine learning library
- Flask team for the web framework
- All contributors to this project

## 📊 Additional Resources

- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Detailed guide for using models
- [Data Dictionary Viewer](github_pages_site/index.html) - Interactive NSCISC variable explorer
- Research PDF - Comprehensive analysis with all figures and insights

---

**Disclaimer**: These models are for research and educational purposes. Clinical decisions should be made by qualified healthcare professionals using comprehensive patient assessment.
