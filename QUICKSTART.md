# 🚀 Quick Start Guide - SepsisSense

Welcome to SepsisSense! This guide will get you up and running with the AI-powered sepsis detection system in minutes.

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Windows/macOS/Linux

## ⚡ Installation

### 1. Clone/Download the Project
```bash
# If using git
git clone https://github.com/yourusername/SepsisSense.git
cd SepsisSense

# Or download and extract the ZIP file
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Demo Data (Optional)
```bash
python test_demo.py
```

## 🎯 Quick Demo

### Option 1: Web Application (Recommended)
```bash
streamlit run app/streamlit_app.py
```
Then open http://localhost:8501 in your browser

### Option 2: Jupyter Notebooks
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Option 3: Python Script
```python
from src.preprocess import create_sample_dataset
from src.train_model import SepsisModelTrainer

# Create sample data
data = create_sample_dataset(n_patients=100)

# Train model
trainer = SepsisModelTrainer()
results = trainer.train_and_evaluate_pipeline(data)

print(f"Model AUC: {results['metrics']['roc_auc']:.3f}")
```

## 🎮 Using the Web Application

1. **Dashboard**: Overview of model performance and system status
2. **Predict Risk**: Enter patient data to get sepsis risk prediction
3. **Analytics**: Explore model performance and feature importance
4. **About**: Learn more about the system

### Sample Patient Entry
- **Patient ID**: P001
- **Age**: 65
- **Gender**: Male
- **Heart Rate**: 95 bpm
- **Blood Pressure**: 110/70 mmHg
- **Temperature**: 37.5°C
- **O2 Saturation**: 94%

Click "Predict Sepsis Risk" to get AI analysis!

## 🔧 Troubleshooting

### Common Issues

**Import Error**: 
```bash
# Ensure you're in the project directory
cd SepsisSense
python -c "import sys; print(sys.path)"
```

**Missing Dependencies**:
```bash
pip install --upgrade -r requirements.txt
```

**Streamlit Issues**:
```bash
streamlit --version
# If outdated: pip install --upgrade streamlit
```

**Memory Issues**:
- Reduce sample size in create_sample_dataset()
- Use fewer patients in demo generation

## 📊 Expected Results

- **Model Performance**: AUC ≥ 0.85
- **Prediction Time**: <1 second per patient
- **Web App**: Loads in 10-30 seconds
- **Feature Engineering**: 100+ features generated

## 🎯 Next Steps

1. **Explore Notebooks**: Start with `01_eda.ipynb` for data exploration
2. **Custom Data**: Replace demo data with your own dataset
3. **Model Tuning**: Adjust hyperparameters in `src/train_model.py`
4. **Deployment**: Use the web app as a foundation for production deployment

## 📞 Need Help?

- **Documentation**: Check the README.md for detailed information
- **Issues**: Common problems and solutions
- **Features**: Explore all available functionalities

Happy predicting! 🩺✨