#!/usr/bin/env python3
"""
Enhanced ML Data Creation
Adds equilibrium_temp and insolation parameters to ML-ready dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_enhanced_ml_data():
    """Create enhanced ML-ready data with additional parameters"""
    
    print("🚀 Creating Enhanced ML-Ready Dataset")
    print("=" * 50)
    
    # Load processed data
    print("Loading processed exoplanet data...")
    data = pd.read_csv('processed_exoplanet_data.csv')
    
    print(f"Original dataset shape: {data.shape}")
    print(f"Available columns: {list(data.columns)}")
    
    # Check data quality for new parameters
    print(f"\nData quality check:")
    print(f"equilibrium_temp - Missing: {data['equilibrium_temp'].isna().sum()} ({data['equilibrium_temp'].isna().mean()*100:.1f}%)")
    print(f"insolation - Missing: {data['insolation'].isna().sum()} ({data['insolation'].isna().mean()*100:.1f}%)")
    
    # Select features for ML (including new parameters)
    feature_columns = [
        'orbital_period', 
        'transit_duration', 
        'planet_radius', 
        'stellar_temp', 
        'stellar_radius', 
        'transit_depth',
        'equilibrium_temp',  # NEW PARAMETER
        'insolation'         # NEW PARAMETER
    ]
    
    print(f"\nSelected features: {feature_columns}")
    
    # Create feature matrix
    X = data[feature_columns].copy()
    y = data['disposition'].copy()
    
    print(f"\nBefore cleaning:")
    print(f"X shape: {X.shape}")
    print(f"Missing values per column:")
    print(X.isna().sum())
    
    # Remove rows with any missing values in selected features
    complete_mask = X.notna().all(axis=1)
    X_clean = X[complete_mask]
    y_clean = y[complete_mask]
    
    print(f"\nAfter removing rows with missing values:")
    print(f"X_clean shape: {X_clean.shape}")
    print(f"Removed {X.shape[0] - X_clean.shape[0]} rows ({(X.shape[0] - X_clean.shape[0])/X.shape[0]*100:.1f}%)")
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(y_clean.value_counts())
    print(f"Class percentages:")
    print(y_clean.value_counts(normalize=True) * 100)
    
    # Create enhanced ML dataset
    enhanced_ml_data = pd.concat([X_clean, y_clean], axis=1)
    
    # Save enhanced ML data
    output_path = 'ml_ready_data_enhanced.csv'
    enhanced_ml_data.to_csv(output_path, index=False)
    
    print(f"\n✅ Enhanced ML dataset saved: {output_path}")
    print(f"Final shape: {enhanced_ml_data.shape}")
    print(f"Features: {list(X_clean.columns)}")
    
    # Data visualization
    create_visualizations(X_clean, y_clean, feature_columns)
    
    return enhanced_ml_data, X_clean, y_clean

def create_visualizations(X, y, feature_columns):
    """Create visualizations for the enhanced dataset"""
    
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    
    # Class distribution
    y.value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Feature distributions
    for i, feature in enumerate(feature_columns):
        row = (i + 1) // 4
        col = (i + 1) % 4
        
        if row < 2 and col < 4:
            axes[row, col].hist(X[feature].dropna(), bins=50, alpha=0.7)
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Count')
    
    # Correlation heatmap
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1, 3], cbar_kws={'shrink': 0.8})
    axes[1, 3].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('assets/data_overview_enhanced.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # New parameters analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Equilibrium temperature vs class
    for class_name in y.unique():
        mask = y == class_name
        axes[0].hist(X[mask]['equilibrium_temp'].dropna(), alpha=0.7, label=class_name, bins=30)
    axes[0].set_title('Equilibrium Temperature by Class')
    axes[0].set_xlabel('Equilibrium Temperature (K)')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    
    # Insolation vs class
    for class_name in y.unique():
        mask = y == class_name
        axes[1].hist(X[mask]['insolation'].dropna(), alpha=0.7, label=class_name, bins=30)
    axes[1].set_title('Insolation by Class')
    axes[1].set_xlabel('Insolation (Earth flux)')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    
    # Scatter plot: equilibrium_temp vs insolation
    scatter = axes[2].scatter(X['equilibrium_temp'], X['insolation'], 
                             c=pd.Categorical(y).codes, alpha=0.6, cmap='viridis')
    axes[2].set_xlabel('Equilibrium Temperature (K)')
    axes[2].set_ylabel('Insolation (Earth flux)')
    axes[2].set_title('Equilibrium Temp vs Insolation')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('assets/new_parameters_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_datasets():
    """Compare original vs enhanced datasets"""
    
    print("\n📊 DATASET COMPARISON")
    print("=" * 50)
    
    # Load original ML data
    original = pd.read_csv('ml_ready_data.csv')
    
    # Load enhanced ML data
    enhanced = pd.read_csv('ml_ready_data_enhanced.csv')
    
    print(f"Original dataset:")
    print(f"  Shape: {original.shape}")
    print(f"  Features: {list(original.columns)}")
    
    print(f"\nEnhanced dataset:")
    print(f"  Shape: {enhanced.shape}")
    print(f"  Features: {list(enhanced.columns)}")
    
    print(f"\nNew parameters added:")
    new_params = set(enhanced.columns) - set(original.columns)
    print(f"  {list(new_params)}")
    
    print(f"\nData loss due to missing values:")
    data_loss = (original.shape[0] - enhanced.shape[0]) / original.shape[0] * 100
    print(f"  {data_loss:.1f}% of data removed")
    
    return original, enhanced

if __name__ == "__main__":
    # Create enhanced dataset
    enhanced_data, X, y = create_enhanced_ml_data()
    
    # Compare with original
    original, enhanced = compare_datasets()
    
    print(f"\n🎯 ENHANCED DATASET READY!")
    print(f"New parameters: equilibrium_temp, insolation")
    print(f"Total features: {len(enhanced.columns) - 1} (was {len(original.columns) - 1})")
    print(f"Ready for model training!")
