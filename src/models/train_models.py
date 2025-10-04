#!/usr/bin/env python3
"""
NASA Exoplanet Detection - Machine Learning Models
Implements Stacking Ensemble approach based on research papers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_ml_data():
    """Load preprocessed ML-ready data"""
    print("Loading ML-ready data...")
    
    data = pd.read_csv('ml_ready_data.csv')
    
    # Separate features and target
    X = data.drop('disposition', axis=1)
    y = data['disposition']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Class distribution:")
    print(y.value_counts())
    print(f"Class percentages:")
    print(y.value_counts(normalize=True) * 100)
    
    return X, y

def prepare_data(X, y):
    """Prepare data for machine learning"""
    
    # Encode labels for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
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
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

def create_base_models():
    """Create base models for stacking ensemble"""
    
    # Base models (as recommended by research papers)
    base_models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        
        'extra_trees': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
    }
    
    return base_models

def create_stacking_ensemble(base_models):
    """Create stacking ensemble model"""
    
    # Convert base_models dict to list of tuples for StackingClassifier
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Stacking classifier with Logistic Regression as meta-learner
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,  # 5-fold cross-validation
        n_jobs=-1
    )
    
    return stacking_model

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, label_encoder):
    """Comprehensive model evaluation"""
    
    print(f"\n{'='*50}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*50}")
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
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
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }
    
    return model, results

def plot_results(results_list):
    """Plot model comparison results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    models = [r['model_name'] for r in results_list]
    test_acc = [r['test_accuracy'] for r in results_list]
    cv_acc = [r['cv_mean'] for r in results_list]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, test_acc, width, label='Test Accuracy', alpha=0.8)
    axes[0, 0].bar(x + width/2, cv_acc, width, label='CV Accuracy', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1-Score comparison (macro average)
    f1_scores = []
    for r in results_list:
        f1_macro = np.mean(r['f1_score'])
        f1_scores.append(f1_macro)
    
    axes[0, 1].bar(models, f1_scores, alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('F1-Score (Macro Avg)')
    axes[0, 1].set_title('Model F1-Score Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion matrix for best model (last one - stacking)
    best_cm = results_list[-1]['confusion_matrix']
    classes = ['CANDIDATE', 'CONFIRMED', 'FALSE_POSITIVE']
    
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {results_list[-1]["model_name"]}')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Per-class performance for best model
    classes_short = ['CAND', 'CONF', 'FP']
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    best_results = results_list[-1]
    x = np.arange(len(classes_short))
    width = 0.25
    
    axes[1, 1].bar(x - width, best_results['precision'], width, label='Precision', alpha=0.8)
    axes[1, 1].bar(x, best_results['recall'], width, label='Recall', alpha=0.8)
    axes[1, 1].bar(x + width, best_results['f1_score'], width, label='F1-Score', alpha=0.8)
    
    axes[1, 1].set_xlabel('Classes')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title(f'Per-Class Performance - {best_results["model_name"]}')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(classes_short)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_models(models_dict, scaler, label_encoder):
    """Save trained models, scaler, and label encoder"""
    
    print(f"\n💾 Saving models...")
    
    # Save each model
    for name, model in models_dict.items():
        filename = f'model_{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, filename)
        print(f"✅ Saved: {filename}")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print(f"✅ Saved: scaler.pkl")
    
    # Save label encoder
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print(f"✅ Saved: label_encoder.pkl")

def main():
    """Main training pipeline"""
    
    print("🚀 NASA Exoplanet Detection - Model Training")
    print("=" * 60)
    
    # Load data
    X, y = load_ml_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_data(X, y)
    
    # Create models
    base_models = create_base_models()
    stacking_model = create_stacking_ensemble(base_models)
    
    # Train and evaluate all models
    all_models = {**base_models, 'Stacking Ensemble': stacking_model}
    
    results_list = []
    trained_models = {}
    
    for name, model in all_models.items():
        trained_model, results = evaluate_model(
            model, X_train, X_test, y_train, y_test, name, label_encoder
        )
        results_list.append(results)
        trained_models[name] = trained_model
    
    # Plot results
    plot_results(results_list)
    
    # Save models
    save_models(trained_models, scaler, label_encoder)
    
    # Summary
    print(f"\n🎯 TRAINING SUMMARY")
    print("=" * 60)
    for r in results_list:
        print(f"{r['model_name']:20s}: {r['test_accuracy']:.4f} accuracy")
    
    best_model = max(results_list, key=lambda x: x['test_accuracy'])
    print(f"\n🏆 Best Model: {best_model['model_name']} ({best_model['test_accuracy']:.4f})")
    
    return trained_models, results_list, scaler, label_encoder

if __name__ == "__main__":
    trained_models, results_list, scaler, label_encoder = main()
