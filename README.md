# ExoTic Vision - NASA Exoplanet Detection Platform

AI-powered web application for automated exoplanet detection using NASA mission data from Kepler, K2, and TESS.

## Performance
- **Accuracy:** 88.21%
- **ROC-AUC:** 0.9448
- **Dataset:** 21,271 NASA records from Kepler, TESS, and K2 missions
- **Model:** Binary Classification (CONFIRMED vs FALSE_POSITIVE)

## Key Features

### Advanced Analytics Dashboard
- Real-time model performance metrics with dynamic updates
- Interactive model comparison charts and scatter plots
- Live feature importance analysis with dynamic model switching
- Dynamic dataset statistics (NASA vs User data composition)
- ROC curves and confusion matrices visualization
- Model performance evolution tracking
- Ensemble component analysis

### Intelligent Data Processing
- **Automatic NASA Format Detection:** Recognizes TESS, Kepler, and K2 dataset formats
- **Smart Column Mapping:** Auto-maps NASA columns to standard features
- **Multi-format Support:** CSV, Excel, and raw NASA datasets
- **Robust Error Handling:** Multiple encoding support (UTF-8, Latin-1, CP1252)
- **Data Validation:** Automatic missing value handling and outlier detection
- **Batch Processing:** Upload and process thousands of records simultaneously

### Model Management System
- **Hybrid Storage:** Local Storage + File System integration
- **Dynamic Model Switching:** Seamless model activation across all pages
- **User-trained Model Support:** Save and manage custom models
- **Model Comparison:** Side-by-side performance analysis
- **Model Download:** Export trained models as .pkl files
- **Session Persistence:** Models persist across browser sessions

### Advanced Training & Optimization
- **Interactive Hyperparameter Tuning:** Real-time sliders for all parameters
- **Method-specific Defaults:** Automatic parameter reset when switching algorithms
- **NASA Data Integration:** Optional combination with NASA datasets
- **Real-time Training Progress:** Live progress bars and status updates
- **Multiple Algorithms:** Random Forest, XGBoost, Extra Trees, Stacking Ensemble
- **Model Validation:** Automatic train-test split and performance metrics

### Professional User Interface
- **Single Prediction Interface:** Instant exoplanet classification
- **Batch Upload with Preview:** Upload CSV files with data preview
- **Results Visualization:** Interactive charts and confidence analysis
- **Export Capabilities:** Download predictions and model files
- **Responsive Design:** Works on desktop and mobile devices
- **Professional Theme:** Clean, scientific interface suitable for NASA researchers


## Quick Start

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
streamlit run src/web_app/main.py
```

### 3. Access Application
Open browser to: `http://localhost:8501`

## Data Sources

- **Kepler Objects of Interest (KOI):** 9,564 confirmed exoplanets and candidates
- **TESS Objects of Interest (TOI):** 7,703 exoplanet candidates
- **K2 Planets and Candidates:** 4,138 planets and candidates

## Model Details

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

## Web Application Features

### Implemented Features
- Single exoplanet prediction
- Batch CSV file upload
- Advanced analytics dashboard
- Model management system
- Hyperparameter tuning
- Real-time model training
- Local Storage integration
- Interactive visualizations
- NASA data auto-detection

## Results Comparison

| Classification Type | Best Model | Accuracy | Improvement |
|-------------------|------------|----------|-------------|
| Ternary (3 classes) | XGBoost | 64.82% | Baseline |
| Binary (2 classes) | Stacking | 88.21% | +23.39% |

## NASA Space Apps Challenge Compliance

- AI/ML model trained on NASA datasets (Kepler, K2, TESS)
- Web interface for user interaction
- High accuracy exoplanet identification (88.21%)
- Real-time data analysis and visualization
- Professional presentation and documentation
- Automated classification system
- Model training and optimization tools

## Usage Instructions

1. **Single Prediction:** Enter planetary and stellar parameters for instant classification
2. **Batch Upload:** Upload CSV files for bulk exoplanet analysis
3. **Analytics Dashboard:** View real-time model performance and comparisons
4. **Model Management:** Switch between models and manage your trained models
5. **Training:** Train new models with custom hyperparameters and NASA data

## Predictions

The system analyzes input parameters and predicts:
- **CONFIRMED:** High probability genuine exoplanet
- **FALSE_POSITIVE:** Likely false signal or noise

## References

- NASA Exoplanet Archive datasets
- MDPI Electronics research paper on exoplanet detection
- MNRAS journal ensemble methods study

## Team

**Nebulatic** - Developing tools that make space exploration more efficient and accessible for researchers worldwide.

---

*Built for NASA Space Apps Challenge 2025*

