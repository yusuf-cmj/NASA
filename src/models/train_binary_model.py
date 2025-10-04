#!/usr/bin/env python3
"""
NASA Exoplanet Detection - Binary Classification
CONFIRMED vs FALSE_POSITIVE (Candidates excluded for higher accuracy)
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_binary_data():
    """Load and filter data for binary classification"""
    print("Loading ML-ready data for binary classification...")
    
    data = pd.read_csv('ml_ready_data.csv')
    
    # Filter out CANDIDATE - keep only CONFIRMED and FALSE_POSITIVE
    binary_data = data[data['disposition'].isin(['CONFIRMED', 'FALSE_POSITIVE'])].copy()
    
    print(f"Original dataset: {data.shape[0]} records")
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

def prepare_binary_data(X, y):
    """Prepare data for binary classification"""
    
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

def create_binary_models():
    """Create optimized models for binary classification"""
    
    # Optimized models for binary classification
    binary_models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,  # More trees for binary
            max_depth=15,      # Deeper trees
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        ),
        
        'extra_trees': ExtraTreesClassifier(
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
            scale_pos_weight=1  # Will adjust based on class ratio
        )
    }
    
    return binary_models

def create_binary_stacking(base_models):
    """Create stacking ensemble for binary classification"""
    
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

def evaluate_binary_model(model, X_train, X_test, y_train, y_test, model_name, label_encoder):
    """Comprehensive binary model evaluation"""
    
    print(f"\n{'='*50}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*50}")
    
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

def plot_binary_results(results_list, label_encoder):
    """Plot binary classification results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Accuracy comparison
    models = [r['model_name'] for r in results_list]
    test_acc = [r['test_accuracy'] for r in results_list]
    cv_acc = [r['cv_mean'] for r in results_list]
    roc_aucs = [r['roc_auc'] if r['roc_auc'] else 0 for r in results_list]
    
    x = np.arange(len(models))
    width = 0.25
    
    axes[0, 0].bar(x - width, test_acc, width, label='Test Accuracy', alpha=0.8)
    axes[0, 0].bar(x, cv_acc, width, label='CV Accuracy', alpha=0.8)
    axes[0, 0].bar(x + width, roc_aucs, width, label='ROC-AUC', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Binary Classification Performance')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # F1-Score comparison
    f1_scores = [np.mean(r['f1_score']) for r in results_list]
    
    axes[0, 1].bar(models, f1_scores, alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('F1-Score (Macro Avg)')
    axes[0, 1].set_title('F1-Score Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROC Curves
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    for r in results_list:
        if r['y_test_proba'] is not None:
            # We need y_test for ROC curve - let's use the last result's test set
            # This is a simplification - in practice, we'd store y_test in results
            pass
    
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].set_title('ROC Curves')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Confusion matrix for best model
    best_model_idx = np.argmax([r['test_accuracy'] for r in results_list])
    best_cm = results_list[best_model_idx]['confusion_matrix']
    classes = ['FALSE_POSITIVE', 'CONFIRMED']
    
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {results_list[best_model_idx]["model_name"]}')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Per-class performance for best model
    best_results = results_list[best_model_idx]
    classes_short = ['FP', 'CONF']
    
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
    
    # Accuracy improvement comparison
    # Compare with ternary results (if available)
    ternary_accuracies = [0.6415, 0.6006, 0.6482, 0.6463]  # From previous run
    binary_accuracies = [r['test_accuracy'] for r in results_list]
    
    improvements = [b - t for b, t in zip(binary_accuracies, ternary_accuracies)]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[1, 2].bar(models, improvements, color=colors, alpha=0.7)
    axes[1, 2].set_xlabel('Models')
    axes[1, 2].set_ylabel('Accuracy Improvement')
    axes[1, 2].set_title('Binary vs Ternary Classification')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('binary_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_binary_models(models_dict, scaler, label_encoder):
    """Save binary classification models"""
    
    print(f"\n💾 Saving binary models...")
    
    # Save each model with binary prefix
    for name, model in models_dict.items():
        filename = f'binary_model_{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, filename)
        print(f"✅ Saved: {filename}")
    
    # Save scaler and encoder
    joblib.dump(scaler, 'binary_scaler.pkl')
    joblib.dump(label_encoder, 'binary_label_encoder.pkl')
    print(f"✅ Saved: binary_scaler.pkl")
    print(f"✅ Saved: binary_label_encoder.pkl")

def main():
    """Main binary classification pipeline"""
    
    print("🚀 NASA Exoplanet Detection - Binary Classification")
    print("🎯 CONFIRMED vs FALSE_POSITIVE (Higher Accuracy Target)")
    print("=" * 70)
    
    # Load binary data
    X, y = load_binary_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_binary_data(X, y)
    
    # Create models
    base_models = create_binary_models()
    stacking_model = create_binary_stacking(base_models)
    
    # Train and evaluate all models
    all_models = {**base_models, 'Binary Stacking': stacking_model}
    
    results_list = []
    trained_models = {}
    
    for name, model in all_models.items():
        trained_model, results = evaluate_binary_model(
            model, X_train, X_test, y_train, y_test, name, label_encoder
        )
        results_list.append(results)
        trained_models[name] = trained_model
    
    # Plot results
    plot_binary_results(results_list, label_encoder)
    
    # Save models
    save_binary_models(trained_models, scaler, label_encoder)
    
    # Summary
    print(f"\n🎯 BINARY CLASSIFICATION SUMMARY")
    print("=" * 70)
    for r in results_list:
        roc_str = f" (ROC-AUC: {r['roc_auc']:.4f})" if r['roc_auc'] else ""
        print(f"{r['model_name']:20s}: {r['test_accuracy']:.4f} accuracy{roc_str}")
    
    best_model = max(results_list, key=lambda x: x['test_accuracy'])
    print(f"\n🏆 Best Binary Model: {best_model['model_name']} ({best_model['test_accuracy']:.4f})")
    
    # Compare with ternary
    print(f"\n📈 IMPROVEMENT vs TERNARY:")
    ternary_best = 0.6482  # XGBoost from previous run
    binary_best = best_model['test_accuracy']
    improvement = binary_best - ternary_best
    print(f"Ternary Best: {ternary_best:.4f}")
    print(f"Binary Best:  {binary_best:.4f}")
    print(f"Improvement:  +{improvement:.4f} ({improvement*100:.2f}%)")
    
    return trained_models, results_list, scaler, label_encoder

if __name__ == "__main__":
    trained_models, results_list, scaler, label_encoder = main()
