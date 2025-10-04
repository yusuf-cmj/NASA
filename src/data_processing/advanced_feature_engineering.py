#!/usr/bin/env python3
"""
Advanced Feature Engineering for 8-Parameter Exoplanet Detection
Creates enhanced features and performs advanced preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.feature_selection import f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_data():
    """Load the 8-parameter enhanced dataset"""
    print("🚀 Loading Enhanced 8-Parameter Dataset")
    print("=" * 50)
    
    data = pd.read_csv('data/processed/ml_ready_data_enhanced.csv')
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {list(data.columns)}")
    print("Class distribution:")
    print(data['disposition'].value_counts())
    print("Class percentages:")
    print(data['disposition'].value_counts(normalize=True) * 100)
    
    return data

def create_advanced_features(df):
    """Create advanced derived features from the 8 parameters"""
    
    print("\n🔧 Creating Advanced Features")
    print("=" * 40)
    
    # Create a copy to avoid modifying original
    enhanced_df = df.copy()
    
    # 1. Physical ratio features
    enhanced_df['planet_star_radius_ratio'] = enhanced_df['planet_radius'] / enhanced_df['stellar_radius']
    enhanced_df['transit_efficiency'] = enhanced_df['transit_depth'] / (enhanced_df['planet_star_radius_ratio'] ** 2)
    
    # 2. Habitability indicators
    enhanced_df['habitability_index'] = enhanced_df['equilibrium_temp'] / 300  # Earth-like temp = 300K
    enhanced_df['insolation_log'] = np.log10(enhanced_df['insolation'] + 1)  # Log transform for skewed data
    
    # 3. Stellar characteristics
    enhanced_df['stellar_density'] = enhanced_df['stellar_radius'] ** 3  # Proportional to stellar volume
    enhanced_df['stellar_luminosity_proxy'] = (enhanced_df['stellar_temp'] / 5778) ** 4 * (enhanced_df['stellar_radius'] ** 2)
    
    # 4. Orbital characteristics
    enhanced_df['orbital_frequency'] = 1 / enhanced_df['orbital_period']  # Frequency in cycles per day
    enhanced_df['transit_probability'] = enhanced_df['stellar_radius'] / enhanced_df['orbital_period']  # Simplified transit probability
    
    # 5. Planet characteristics
    enhanced_df['planet_volume'] = (4/3) * np.pi * (enhanced_df['planet_radius'] ** 3)
    enhanced_df['planet_density_proxy'] = enhanced_df['planet_radius'] ** 2  # Surface area proxy
    
    # 6. Temperature and flux relationships
    enhanced_df['temp_insolation_ratio'] = enhanced_df['equilibrium_temp'] / (enhanced_df['insolation'] + 1)
    enhanced_df['stellar_flux_density'] = enhanced_df['insolation'] / (enhanced_df['stellar_radius'] ** 2)
    
    # 7. Transit characteristics
    enhanced_df['transit_duration_ratio'] = enhanced_df['transit_duration'] / enhanced_df['orbital_period']
    enhanced_df['transit_depth_normalized'] = enhanced_df['transit_depth'] / enhanced_df['stellar_radius']
    
    # 8. Combined physical indicators
    enhanced_df['habitability_score'] = (
        (enhanced_df['habitability_index'] > 0.5) & 
        (enhanced_df['habitability_index'] < 2.0) &
        (enhanced_df['insolation'] > 0.1) &
        (enhanced_df['insolation'] < 10)
    ).astype(int)
    
    # 9. Size classification
    enhanced_df['planet_size_class'] = pd.cut(
        enhanced_df['planet_radius'], 
        bins=[0, 1.25, 2, 4, 6.25, 10, np.inf],
        labels=['Sub-Earth', 'Earth-like', 'Super-Earth', 'Neptune-like', 'Jupiter-like', 'Giant']
    )
    
    # 10. Temperature classification
    enhanced_df['temp_class'] = pd.cut(
        enhanced_df['equilibrium_temp'],
        bins=[0, 200, 300, 500, 1000, np.inf],
        labels=['Cold', 'Temperate', 'Warm', 'Hot', 'Very Hot']
    )
    
    print(f"Created {len(enhanced_df.columns) - len(df.columns)} new features")
    print(f"Total features now: {len(enhanced_df.columns)}")
    
    return enhanced_df

def analyze_feature_importance(X, y, feature_names):
    """Analyze feature importance using multiple methods"""
    
    print("\n📊 Analyzing Feature Importance")
    print("=" * 40)
    
    # Method 1: F-test
    f_scores, f_pvalues = f_classif(X, y)
    f_importance = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'f_pvalue': f_pvalues
    }).sort_values('f_score', ascending=False)
    
    # Method 2: Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Combine results
    importance_df = f_importance.merge(mi_importance, on='feature')
    importance_df['combined_score'] = (
        importance_df['f_score'] / importance_df['f_score'].max() + 
        importance_df['mi_score'] / importance_df['mi_score'].max()
    ) / 2
    
    print("Top 10 Most Important Features:")
    print(importance_df.head(10)[['feature', 'f_score', 'mi_score', 'combined_score']])
    
    return importance_df

def apply_transformations(X, y, feature_names):
    """Apply various transformations to improve model performance"""
    
    print("\n🔄 Applying Data Transformations")
    print("=" * 40)
    
    # 1. Log transformation for highly skewed features
    skewed_features = []
    for i, feature in enumerate(feature_names):
        if X[:, i].min() > 0:  # Only for positive values
            skewness = abs(pd.Series(X[:, i]).skew())
            if skewness > 1.0:  # Highly skewed
                skewed_features.append(i)
                X[:, i] = np.log1p(X[:, i])  # log(1+x) to handle zeros
    
    print(f"Applied log transformation to {len(skewed_features)} features")
    
    # 2. Power transformation for remaining skewed features
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    x_transformed = pt.fit_transform(X)
    
    print("Applied Yeo-Johnson power transformation")
    
    return x_transformed, pt

def create_feature_visualizations(_, __):
    """Create comprehensive visualizations for feature analysis"""
    
    print("\n📈 Feature Analysis Complete")
    print("=" * 40)
    
    # Skip visualization - focus on data processing
    print("Skipping visualizations for faster processing")

def prepare_ml_data(df, use_advanced_features=True):
    """Prepare data for machine learning with optional advanced features"""
    
    print("\n🤖 Preparing ML Data")
    print("=" * 30)
    
    if use_advanced_features:
        # Use all features including advanced ones
        feature_columns = [col for col in df.columns if col not in ['disposition', 'object_id', 'source']]
        print(f"Using {len(feature_columns)} features (including advanced features)")
    else:
        # Use only original 8 parameters
        feature_columns = [
            'orbital_period', 'transit_duration', 'planet_radius', 
            'stellar_temp', 'stellar_radius', 'transit_depth',
            'equilibrium_temp', 'insolation'
        ]
        print(f"Using {len(feature_columns)} original features")
    
    # Create feature matrix
    X = df[feature_columns].copy()
    y = df['disposition'].copy()
    
    # Handle missing values
    print("Missing values before cleaning:")
    print(X.isnull().sum().sum())
    
    # Remove rows with any missing values
    complete_mask = X.notna().all(axis=1)
    x_clean = X[complete_mask]
    y_clean = y[complete_mask]
    
    print(f"After cleaning: {x_clean.shape[0]} samples")
    print(f"Removed {X.shape[0] - x_clean.shape[0]} samples with missing values")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_clean)
    
    print("Class distribution:")
    print(pd.Series(y_encoded).value_counts())
    
    return x_clean, y_encoded, feature_columns, label_encoder

def main():
    """Main feature engineering pipeline"""
    
    print("🚀 Advanced Feature Engineering Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_enhanced_data()
    
    # Create advanced features
    enhanced_df = create_advanced_features(df)
    
    # Prepare ML data with advanced features
    x_advanced, y_advanced, feature_names_advanced, _ = prepare_ml_data(enhanced_df, use_advanced_features=True)
    
    # Prepare ML data with original features only
    x_original, y_original, feature_names_original, _ = prepare_ml_data(enhanced_df, use_advanced_features=False)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(x_original, y_original, feature_names_original)
    
    # Create visualizations
    create_feature_visualizations(enhanced_df, feature_names_advanced)
    
    
    # Save feature importance
    importance_df.to_csv('data/processed/feature_importance.csv', index=False)
    print("✅ Saved feature importance: data/processed/feature_importance.csv")
    
    print("\n🎯 FEATURE ENGINEERING COMPLETE!")
    print(f"Original features: {len(feature_names_original)}")
    print(f"Advanced features: {len(feature_names_advanced)}")
    print("Ready for advanced model training!")
    
    return enhanced_df, x_advanced, y_advanced, x_original, y_original, feature_names_advanced, feature_names_original

if __name__ == "__main__":
    enhanced_df, x_advanced, y_advanced, x_original, y_original, feature_names_advanced, feature_names_original = main()
