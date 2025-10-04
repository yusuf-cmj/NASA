#!/usr/bin/env python3
"""
Enhanced NASA Exoplanet Detection - Binary Classification
CONFIRMED vs FALSE_POSITIVE with 8 parameters
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_binary_data():
    """Load and filter enhanced data for binary classification"""
    print("Loading enhanced ML-ready data for binary classification...")
    
    data = pd.read_csv('data/processed/ml_ready_data_enhanced.csv')
    
    # Filter out CANDIDATE - keep only CONFIRMED and FALSE_POSITIVE
    binary_data = data[data['disposition'].isin(['CONFIRMED', 'FALSE_POSITIVE'])].copy()
    
    print(f"Enhanced dataset: {data.shape[0]} records")
    print(f"Binary dataset: {binary_data.shape[0]} records")
    print(f"Removed CANDIDATES: {data.shape[0] - binary_data.shape[0]} records")
    
    # Separate features and target
    X = binary_data.drop('disposition', axis=1)
    y = binary_data['disposition']
    
    print(f"\nBinary dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Class distribution:")
    print(y.value_counts())
    print(f"Class percentages:")
    print(y.value_counts(normalize=True) * 100)
    
    return X, y

def prepare_enhanced_binary_data(X, y):
    """Prepare enhanced data for binary classification"""
    
    # Encode labels (CONFIRMED=1, FALSE_POSITIVE=0)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Train-test split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

def create_enhanced_binary_models():
    """Create optimized models for enhanced binary classification with LightGBM and CatBoost"""
    
    # Enhanced models for binary classification including LightGBM and CatBoost
    binary_models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        
        
        'xgboost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=1
        ),
        
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.001,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            class_weight='balanced',
            verbose=-1,  # Suppress output
            n_jobs=-1
        ),
        
        'catboost': CatBoostClassifier(
            iterations=200,
            depth=8,
            learning_rate=0.1,
            colsample_bylevel=0.9,
            random_state=42,
            auto_class_weights='Balanced',
            verbose=False,  # Suppress output
            thread_count=-1
        )
    }
    
    return binary_models

def create_enhanced_binary_stacking(base_models):
    """Create stacking ensemble for enhanced binary classification"""
    
    estimators = [(name, model) for name, model in base_models.items()]
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        ),
        cv=5,
        n_jobs=-1
    )
    
    return stacking_model

def evaluate_enhanced_binary_model(model, X_train, X_test, y_train, y_test, model_name, label_encoder):
    """Comprehensive enhanced binary model evaluation"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING ENHANCED BINARY: {model_name}")
    print(f"{'='*60}")
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Prediction probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    else:
        y_test_proba = None
    
    # Accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # ROC-AUC Score
    if y_test_proba is not None:
        roc_auc = roc_auc_score(y_test, y_test_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    else:
        roc_auc = None
    
    # Convert predictions back to original labels for reporting
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    # Detailed classification report
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test_labels, y_test_pred_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_test_pred, average=None)
    
    results = {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'y_test_proba': y_test_proba
    }
    
    return model, results

# Plot function removed - no visual output needed

def save_enhanced_binary_models(models_dict, scaler, label_encoder):
    """Save enhanced binary classification models"""
    
    print(f"\nSaving enhanced binary models...")
    
    # Save each model with enhanced binary prefix
    for name, model in models_dict.items():
        filename = f'data/models/enhanced_binary_model_{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, filename)
        print(f"Saved: {filename}")
    
    # Save scaler and encoder
    joblib.dump(scaler, 'data/models/enhanced_binary_scaler.pkl')
    joblib.dump(label_encoder, 'data/models/enhanced_binary_label_encoder.pkl')
    print(f"Saved: data/models/enhanced_binary_scaler.pkl")
    print(f"Saved: data/models/enhanced_binary_label_encoder.pkl")

def main():
    """Main enhanced binary classification pipeline"""
    
    print("NASA Exoplanet Detection - Enhanced Binary Classification")
    print("CONFIRMED vs FALSE_POSITIVE with 8 parameters")
    print("=" * 70)
    
    # Load enhanced binary data
    X, y = load_enhanced_binary_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_enhanced_binary_data(X, y)
    
    # Create models
    base_models = create_enhanced_binary_models()
    stacking_model = create_enhanced_binary_stacking(base_models)
    
    # Train and evaluate all models
    all_models = {**base_models, 'Enhanced Binary Stacking': stacking_model}
    
    results_list = []
    trained_models = {}
    
    for name, model in all_models.items():
        trained_model, results = evaluate_enhanced_binary_model(
            model, X_train, X_test, y_train, y_test, name, label_encoder
        )
        results_list.append(results)
        trained_models[name] = trained_model
    
    # Plot results (disabled)
    # plot_enhanced_binary_results(results_list, label_encoder)
    
    # Save models
    save_enhanced_binary_models(trained_models, scaler, label_encoder)
    
    # Summary
    print(f"\nENHANCED BINARY CLASSIFICATION SUMMARY")
    print("=" * 70)
    for r in results_list:
        roc_str = f" (ROC-AUC: {r['roc_auc']:.4f})" if r['roc_auc'] else ""
        print(f"{r['model_name']:25s}: {r['test_accuracy']:.4f} accuracy{roc_str}")
    
    best_model = max(results_list, key=lambda x: x['test_accuracy'])
    print(f"\nBest Enhanced Binary Model: {best_model['model_name']} ({best_model['test_accuracy']:.4f})")
    
    return trained_models, results_list, scaler, label_encoder

if __name__ == "__main__":
    trained_models, results_list, scaler, label_encoder = main()

