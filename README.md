# 🚀 NASA Exoplanet Detection System

AI-powered web application for detecting confirmed exoplanets vs false positives using NASA data.

## 📊 Performance
- **Accuracy:** 88.21%
- **ROC-AUC:** 0.9448
- **Dataset:** 21,271 NASA records from Kepler, TESS, and K2 missions

## 🗂️ Project Structure

```
NASA_Exoplanet_Detection/
├── 📁 data/
│   ├── raw/                                    # Original NASA datasets
│   │   ├── cumulative_2025.10.02_09.39.23.csv # TESS TOI data (7,735 rows)
│   │   ├── cumulative_2025.10.02_09.54.58.csv  # Kepler KOI data (9,652 rows)
│   │   └── k2pandc_2025.10.02_10.02.12.csv     # K2 Candidates data (4,138 rows)
│   ├── processed/                              # Cleaned datasets
│   │   ├── processed_exoplanet_data.csv       # Combined & normalized data
│   │   └── ml_ready_data.csv                  # ML training data
│   └── models/                                # Trained model files
│       ├── binary_model_binary_stacking.pkl   # Best performing model
│       ├── binary_model_xgboost.pkl           # XGBoost model
│       ├── binary_model_random_forest.pkl     # Random Forest model
│       ├── binary_scaler.pkl                  # Feature scaler
│       └── binary_label_encoder.pkl           # Label encoder
├── 📁 src/
│   ├── data_processing/
│   │   └── analyze_datasets.py               # Dataset analysis
│   │   └── data_preprocessing.py             # Data cleaning & normalization
│   ├── models/
│   │   └── train_binary_model.py             # Binary classification training
│   │   └── train_models.py                   # Ternary classification (original)
│   └── web_app/
│       └── web_app.py                        # Streamlit web application
├── 📁 assets/
│   ├── data_overview.png                     # EDA visualizations
│   ├── binary_model_comparison.png          # Model performance plots
│   └── model_comparison.png                 # Ternary vs Binary comparison
├── 📄 requirements.txt                       # Python dependencies
├── 📄 README.md                             # This file
└── 📄 .gitignore                           # Git ignore rules
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv nasa_exoplanet_env
nasa_exoplanet_env\Scripts\activate  # Windows
# source nasa_exoplanet_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Web Application
```bash
streamlit run src/web_app/web_app.py
```

### 3. Access Application
Open browser to: `http://localhost:8501`

## 📊 Data Sources

- **Kepler Objects of Interest (KOI):** 9,564 confirmed exoplanets and candidates
- **TESS Objects of Interest (TOI):** 7,703 exoplanet candidates
- **K2 Planets and Candidates:** 4,138 planets and candidates

## 🤖 Model Details

### Binary Classification: CONFIRMED vs FALSE_POSITIVE
- **Method:** Stacking Ensemble (Random Forest + XGBoost + Extra Trees)
- **Features:** 6 key astronomical parameters
- **Performance:** 88.21% accuracy, 0.9448 ROC-AUC

### Key Features:
- Orbital Period (days)
- Transit Duration (hours)
- Planet Radius (Earth radii)
- Stellar Temperature (Kelvin)
- Stellar Radius (Solar radii)
- Transit Depth (parts per million)

## 🌐 Web Application Features

### Phase 1 MVP ✅
- ✅ Single exoplanet prediction
- ✅ Real-time classification
- ✅ Model performance metrics
- ✅ Interactive input forms
- ✅ Confidence intervals

### Phase 2 (Planned)
- 🔄 CSV file upload
- 🔄 Batch processing
- 🔄 Advanced visualizations
- 🔄 Model comparison

## 📈 Results Comparison

| Classification Type | Best Model | Accuracy | Improvement |
|-------------------|------------|----------|-------------|
| Ternary (3 classes) | XGBoost | 64.82% | Baseline |
| Binary (2 classes) | Stacking | 88.21% | +23.39% |

## 🎯 NASA Hackathon Compliance

- ✅ AI/ML model trained on NASA datasets
- ✅ Web interface for user interaction
- ✅ High accuracy exoplanet identification
- ✅ Real-time data analysis
- ✅ Professional presentation

## 👥 Usage Instructions

1. **Single Prediction:** Enter planetary and stellar parameters to get instant prediction
2. **Model Performance:** View detailed accuracy metrics and performance analysis
3. **About:** Learn about the project methodology and data sources

## 🔮 Predictions

The system analyzes input parameters and predicts:
- **✅ CONFIRMED:** High probability genuine exoplanet
- **❌ FALSE_POSITIVE:** Likely false signal or noise

## 📚 References

- NASA Exoplanet Archive datasets
- MDPI Electronics research paper on exoplanet detection
- MNRAS journal ensemble methods study

---

*Built for NASA Hackathon 2025 🚀*

