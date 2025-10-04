#!/usr/bin/env python3
"""
NASA Exoplanet Data Preprocessing and Normalization
Combines TESS, Kepler, and K2 datasets into unified format
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_datasets():
    """Load all three datasets"""
    print("Loading datasets...")
    
    # Load datasets (skip comment lines)
    tess = pd.read_csv("cumulative_2025.10.02_09.39.23.csv", comment='#')
    kepler = pd.read_csv("cumulative_2025.10.02_09.54.58.csv", comment='#')
    k2 = pd.read_csv("k2pandc_2025.10.02_10.02.12.csv", comment='#')
    
    print(f"TESS: {tess.shape}")
    print(f"Kepler: {kepler.shape}")
    print(f"K2: {k2.shape}")
    
    return tess, kepler, k2

def normalize_disposition(df, source):
    """Normalize disposition columns to standard format"""
    
    if source == 'tess':
        # TESS: PC, FP, CP, KP, APC, FA
        disposition_map = {
            'PC': 'CANDIDATE',      # Planet Candidate
            'CP': 'CONFIRMED',      # Confirmed Planet  
            'KP': 'CONFIRMED',      # Known Planet (already confirmed)
            'FP': 'FALSE_POSITIVE', # False Positive
            'APC': 'CANDIDATE',     # Ambiguous Planet Candidate
            'FA': 'FALSE_POSITIVE'  # False Alarm
        }
        df['standard_disposition'] = df['tfopwg_disp'].map(disposition_map)
        
    elif source == 'kepler':
        # Kepler: CONFIRMED, CANDIDATE, FALSE POSITIVE
        disposition_map = {
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE', 
            'FALSE POSITIVE': 'FALSE_POSITIVE'
        }
        df['standard_disposition'] = df['koi_disposition'].map(disposition_map)
        
    elif source == 'k2':
        # K2: CONFIRMED, CANDIDATE, FALSE POSITIVE, REFUTED
        disposition_map = {
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE',
            'FALSE POSITIVE': 'FALSE_POSITIVE',
            'REFUTED': 'FALSE_POSITIVE'  # Refuted = False Positive
        }
        df['standard_disposition'] = df['disposition'].map(disposition_map)
    
    return df

def extract_common_features(tess, kepler, k2):
    """Extract and normalize common features from all datasets"""
    
    # TESS features
    tess_features = pd.DataFrame({
        'source': 'TESS',
        'object_id': tess['toi'].astype(str),
        'orbital_period': tess.get('pl_orbper', np.nan),
        'transit_duration': tess.get('pl_trandurh', np.nan), 
        'planet_radius': tess.get('pl_rade', np.nan),
        'stellar_temp': tess.get('st_teff', np.nan),
        'stellar_radius': tess.get('st_rad', np.nan),
        'stellar_mass': tess.get('st_mass', np.nan),
        'transit_depth': tess.get('pl_trandep', np.nan),
        'equilibrium_temp': tess.get('pl_eqt', np.nan),
        'insolation': tess.get('pl_insol', np.nan),
        'disposition': tess['standard_disposition']
    })
    
    # Kepler features  
    kepler_features = pd.DataFrame({
        'source': 'Kepler',
        'object_id': kepler['kepoi_name'].astype(str),
        'orbital_period': kepler.get('koi_period', np.nan),
        'transit_duration': kepler.get('koi_duration', np.nan),
        'planet_radius': kepler.get('koi_prad', np.nan), 
        'stellar_temp': kepler.get('koi_steff', np.nan),
        'stellar_radius': kepler.get('koi_srad', np.nan),
        'stellar_mass': kepler.get('koi_smass', np.nan),
        'transit_depth': kepler.get('koi_depth', np.nan),
        'equilibrium_temp': kepler.get('koi_teq', np.nan),
        'insolation': kepler.get('koi_insol', np.nan),
        'disposition': kepler['standard_disposition']
    })
    
    # K2 features
    k2_features = pd.DataFrame({
        'source': 'K2', 
        'object_id': k2['pl_name'].astype(str),
        'orbital_period': k2.get('pl_orbper', np.nan),
        'transit_duration': k2.get('pl_trandur', np.nan),
        'planet_radius': k2.get('pl_rade', np.nan),
        'stellar_temp': k2.get('st_teff', np.nan),
        'stellar_radius': k2.get('st_rad', np.nan), 
        'stellar_mass': k2.get('st_mass', np.nan),
        'transit_depth': k2.get('pl_trandep', np.nan),
        'equilibrium_temp': k2.get('pl_eqt', np.nan),
        'insolation': k2.get('pl_insol', np.nan),
        'disposition': k2['standard_disposition']
    })
    
    # Combine all datasets
    combined = pd.concat([tess_features, kepler_features, k2_features], ignore_index=True)
    
    return combined

def clean_data(df):
    """Clean and prepare data for ML"""
    
    print(f"Before cleaning: {df.shape}")
    
    # Remove rows with missing disposition
    df = df.dropna(subset=['disposition'])
    print(f"After removing missing disposition: {df.shape}")
    
    # Remove rows where all key features are missing
    key_features = ['orbital_period', 'transit_duration', 'planet_radius', 'stellar_temp']
    df = df.dropna(subset=key_features, how='all')
    print(f"After removing rows with all missing features: {df.shape}")
    
    # Remove outliers (basic filtering)
    if 'orbital_period' in df.columns:
        df = df[(df['orbital_period'] > 0) & (df['orbital_period'] < 1000)]
    
    if 'planet_radius' in df.columns:
        df = df[(df['planet_radius'] > 0) & (df['planet_radius'] < 50)]
        
    if 'stellar_temp' in df.columns:
        df = df[(df['stellar_temp'] > 2000) & (df['stellar_temp'] < 10000)]
    
    print(f"After outlier removal: {df.shape}")
    
    return df

def create_ml_features(df):
    """Create features for machine learning"""
    
    # Select features for ML (only complete cases for now)
    feature_columns = [
        'orbital_period', 'transit_duration', 'planet_radius', 
        'stellar_temp', 'stellar_radius', 'transit_depth'
    ]
    
    # Create feature matrix
    X = df[feature_columns].copy()
    y = df['disposition'].copy()
    
    # Remove rows with any missing values in selected features
    complete_mask = X.notna().all(axis=1)
    X = X[complete_mask]
    y = y[complete_mask]
    
    print(f"ML Dataset shape: {X.shape}")
    print(f"Class distribution:")
    print(y.value_counts())
    
    return X, y

def main():
    """Main preprocessing pipeline"""
    
    # Load datasets
    tess, kepler, k2 = load_datasets()
    
    # Normalize dispositions
    tess = normalize_disposition(tess, 'tess')
    kepler = normalize_disposition(kepler, 'kepler') 
    k2 = normalize_disposition(k2, 'k2')
    
    # Extract common features
    combined = extract_common_features(tess, kepler, k2)
    
    # Clean data
    combined_clean = clean_data(combined)
    
    # Create ML features
    X, y = create_ml_features(combined_clean)
    
    # Save processed data
    combined_clean.to_csv('processed_exoplanet_data.csv', index=False)
    
    # Save ML-ready data
    ml_data = pd.concat([X, y], axis=1)
    ml_data.to_csv('ml_ready_data.csv', index=False)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"📁 Saved: processed_exoplanet_data.csv ({combined_clean.shape})")
    print(f"📁 Saved: ml_ready_data.csv ({ml_data.shape})")
    
    # Quick visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    y.value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    combined_clean['source'].value_counts().plot(kind='bar')
    plt.title('Data Source Distribution')
    
    plt.subplot(2, 2, 3)
    plt.scatter(X['orbital_period'], X['planet_radius'], alpha=0.5)
    plt.xlabel('Orbital Period (days)')
    plt.ylabel('Planet Radius (Earth radii)')
    plt.title('Period vs Radius')
    
    plt.subplot(2, 2, 4)
    plt.hist(X['stellar_temp'], bins=50, alpha=0.7)
    plt.xlabel('Stellar Temperature (K)')
    plt.ylabel('Count')
    plt.title('Stellar Temperature Distribution')
    
    plt.tight_layout()
    plt.savefig('data_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return combined_clean, X, y

if __name__ == "__main__":
    combined_clean, X, y = main()
