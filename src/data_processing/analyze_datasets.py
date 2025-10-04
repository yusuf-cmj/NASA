#!/usr/bin/env python3
"""
NASA Exoplanet Dataset Analysis
Analyzing the structure and content of the three datasets
"""

import pandas as pd
import numpy as np

def analyze_dataset(filepath, dataset_name):
    """Analyze a single dataset"""
    print(f"\n{'='*50}")
    print(f"ANALYZING: {dataset_name}")
    print(f"File: {filepath}")
    print(f"{'='*50}")
    
    # Read the dataset (skip comment lines starting with #)
    df = pd.read_csv(filepath, comment='#')
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Show first few column names
    print(f"\nFirst 10 columns:")
    for i, col in enumerate(df.columns[:10]):
        print(f"  {i+1:2d}. {col}")
    
    # Look for disposition/classification columns
    print(f"\nLooking for classification columns:")
    classification_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['disp', 'disposition', 'status', 'class']):
            classification_cols.append(col)
            print(f"  ✓ {col}")
    
    # Show unique values in classification columns
    for col in classification_cols:
        if col in df.columns:
            unique_vals = df[col].value_counts()
            print(f"\n  {col} distribution:")
            for val, count in unique_vals.head(10).items():
                print(f"    {val}: {count}")
    
    # Look for key features mentioned in papers
    print(f"\nLooking for key features:")
    key_features = ['period', 'duration', 'radius', 'depth', 'temp', 'teff', 'mass']
    found_features = []
    
    for feature in key_features:
        matching_cols = [col for col in df.columns if feature in col.lower()]
        if matching_cols:
            found_features.extend(matching_cols)
            print(f"  ✓ {feature}: {matching_cols[:3]}")  # Show first 3 matches
    
    # Check for missing values in key columns
    print(f"\nMissing values in key columns:")
    for col in classification_cols + found_features[:5]:  # Check first 5 feature cols
        if col in df.columns:
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            print(f"  {col}: {missing} ({missing_pct:.1f}%)")
    
    return df, classification_cols, found_features

def main():
    """Main analysis function"""
    
    # Dataset file paths
    datasets = [
        ("cumulative_2025.10.02_09.39.23.csv", "TESS Objects of Interest (TOI)"),
        ("cumulative_2025.10.02_09.54.58.csv", "Kepler Objects of Interest (KOI)"), 
        ("k2pandc_2025.10.02_10.02.12.csv", "K2 Planets and Candidates")
    ]
    
    results = {}
    
    # Analyze each dataset
    for filepath, name in datasets:
        try:
            df, class_cols, features = analyze_dataset(filepath, name)
            results[name] = {
                'dataframe': df,
                'classification_columns': class_cols,
                'feature_columns': features,
                'shape': df.shape
            }
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    total_records = 0
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Records: {data['shape'][0]:,}")
        print(f"  Features: {data['shape'][1]}")
        print(f"  Classification cols: {data['classification_columns']}")
        total_records += data['shape'][0]
    
    print(f"\nTOTAL RECORDS: {total_records:,}")
    
    # Suggest normalization strategy
    print(f"\n{'='*60}")
    print("NORMALIZATION STRATEGY")
    print(f"{'='*60}")
    
    print("1. Classification Column Mapping:")
    print("   TESS: 'tfopwg_disp' → standard_disposition")
    print("   Kepler: 'koi_disposition' → standard_disposition") 
    print("   K2: 'disposition' → standard_disposition")
    
    print("\n2. Key Feature Mapping:")
    print("   Orbital Period: toi → pl_orbper, koi → koi_period, k2 → pl_orbper")
    print("   Transit Duration: toi → pl_trandurh, koi → koi_duration, k2 → pl_trandur")
    print("   Planet Radius: toi → pl_rade, koi → koi_prad, k2 → pl_rade")
    print("   Stellar Temp: toi → st_teff, koi → koi_steff, k2 → st_teff")

if __name__ == "__main__":
    main()

