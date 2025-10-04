#!/usr/bin/env python3
"""
Advanced NASA Exoplanet Detection Models
8-Parameter Enhanced Models with Advanced Algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_advanced_data():
    """Load advanced feature-engineered dataset"""
    print("🚀 Loading Advanced 8-Parameter Dataset")
    print("=" * 50)
    
    data = pd.read_csv('data/processed/ml_ready_data_enhanced.csv')
    
    # Filter for binary classification (CONFIRMED vs FALSE_POSITIVE)
    binary_data = data[data['disposition'].isin(['CONFIRMED', 'FALSE_POSITIVE'])].copy()
    
    print(f"Enhanced dataset: {data.shape[0]} records")
    print(f"Binary dataset: {binary_data.shape[0]} records")
    print(f"Removed CANDIDATES: {data.shape[0] - binary_data.shape[0]} records")
    
    # Separate features and target
    X = binary_data.drop('disposition', axis=1)
    y = binary_data['disposition']
    
    print(f"\nBinary dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print("Class distribution:")
    print(y.value_counts())
    print("Class percentages:")
    print(y.value_counts(normalize=True) * 100)
    
    return X, y

def prepare_advanced_data(X, y):
    """Prepare data with advanced preprocessing"""
    
    print("\n🔧 Advanced Data Preprocessing")
    print("=" * 40)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Advanced scaling with PowerTransformer
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=X.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=X.columns)
    
    print("Training set:", x_train_scaled.shape)
    print("Test set:", x_test_scaled.shape)
    
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, label_encoder

def create_advanced_models():
    """Create advanced models with optimized hyperparameters"""
    
    print("\n🤖 Creating Advanced Models")
    print("=" * 40)
    
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        
        'extra_trees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        
        'xgboost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=1
        ),
        
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            class_weight='balanced',
            verbose=-1,
            n_jobs=-1
        ),
        
        'catboost': CatBoostClassifier(
            iterations=300,
            depth=10,
            learning_rate=0.05,
            colsample_bylevel=0.8,
            reg_lambda=0.1,
            random_state=42,
            auto_class_weights='Balanced',
            verbose=False,
            thread_count=-1
        ),
        
        'svm': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    return models

def create_advanced_ensembles(base_models):
    """Create advanced ensemble models"""
    
    print("\n🎯 Creating Advanced Ensembles")
    print("=" * 40)
    
    # Voting Classifier (Hard Voting)
    voting_estimators = [(name, model) for name, model in base_models.items()]
    voting_hard = VotingClassifier(
        estimators=voting_estimators,
        voting='hard'
    )
    
    # Voting Classifier (Soft Voting)
    voting_soft = VotingClassifier(
        estimators=voting_estimators,
        voting='soft'
    )
    
    # Stacking Classifier
    stacking = StackingClassifier(
        estimators=voting_estimators,
        final_estimator=LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        ),
        cv=5,
        n_jobs=-1
    )
    
    # Advanced Stacking with Neural Network meta-learner
    stacking_nn = StackingClassifier(
        estimators=voting_estimators,
        final_estimator=MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=300,
            random_state=42
        ),
        cv=5,
        n_jobs=-1
    )
    
    ensemble_models = {
        'Voting Hard': voting_hard,
        'Voting Soft': voting_soft,
        'Stacking LR': stacking,
        'Stacking NN': stacking_nn
    }
    
    return ensemble_models

def evaluate_advanced_model(model, X_train, X_test, y_train, y_test, model_name, label_encoder):
    """Comprehensive evaluation of advanced models"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING ADVANCED MODEL: {model_name}")
    print(f"{'='*60}")
    
    # Train the model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Prediction probabilities
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = None
    
    # Accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # ROC-AUC Score
    if y_test_proba is not None:
        roc_auc = roc_auc_score(y_test, y_test_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    else:
        roc_auc = None
    
    # Convert predictions back to original labels
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    # Classification report
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test_labels, y_test_pred_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Per-class metrics
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

def plot_advanced_results(results_list, _):
    """Plot comprehensive results for advanced models"""
    
    print("\n📊 Advanced Results Analysis Complete")
    print("=" * 50)
    
    # Skip visualization - focus on model performance
    print("Skipping visualizations for faster processing")
    
    # Create performance dataframe
    performance_data = []
    for r in results_list:
        performance_data.append({
            'Model': r['model_name'],
            'Test Accuracy': r['test_accuracy'],
            'CV Accuracy': r['cv_mean'],
            'ROC-AUC': r['roc_auc'] if r['roc_auc'] else 0,
            'F1-Score': np.mean(r['f1_score'])
        })
    
    perf_df = pd.DataFrame(performance_data)
    perf_df['Overall Score'] = (
        perf_df['Test Accuracy'] * 0.3 + 
        perf_df['CV Accuracy'] * 0.3 + 
        perf_df['ROC-AUC'] * 0.2 + 
        perf_df['F1-Score'] * 0.2
    )
    perf_df = perf_df.sort_values('Overall Score', ascending=False)
    
    return perf_df

def save_advanced_models(models_dict, scaler, label_encoder):
    """Save advanced models and preprocessing objects"""
    
    print("\n💾 Saving Advanced Models")
    print("=" * 40)
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('data/models/advanced', exist_ok=True)
    
    # Save each model
    for name, model in models_dict.items():
        filename = f'data/models/advanced/advanced_model_{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, filename)
        print(f"✅ Saved: {filename}")
    
    # Save scaler and encoder
    joblib.dump(scaler, 'data/models/advanced/advanced_scaler.pkl')
    joblib.dump(label_encoder, 'data/models/advanced/advanced_label_encoder.pkl')
    print("✅ Saved: advanced_scaler.pkl")
    print("✅ Saved: advanced_label_encoder.pkl")

def main():
    """Main advanced model training pipeline"""
    
    print("🚀 NASA Exoplanet Detection - Advanced 8-Parameter Models")
    print("=" * 70)
    
    # Load data
    X, y = load_advanced_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_advanced_data(X, y)
    
    # Create models
    base_models = create_advanced_models()
    ensemble_models = create_advanced_ensembles(base_models)
    
    # Combine all models
    all_models = {**base_models, **ensemble_models}
    
    # Train and evaluate all models
    results_list = []
    trained_models = {}
    
    for name, model in all_models.items():
        trained_model, results = evaluate_advanced_model(
            model, X_train, X_test, y_train, y_test, name, label_encoder
        )
        results_list.append(results)
        trained_models[name] = trained_model
    
    # Plot results
    performance_df = plot_advanced_results(results_list, label_encoder)
    
    # Save models
    save_advanced_models(trained_models, scaler, label_encoder)
    
    # Summary
    print("\n🎯 ADVANCED MODEL TRAINING SUMMARY")
    print("=" * 70)
    for r in results_list:
        roc_str = f" (ROC-AUC: {r['roc_auc']:.4f})" if r['roc_auc'] else ""
        print(f"{r['model_name']:25s}: {r['test_accuracy']:.4f} accuracy{roc_str}")
    
    best_model = max(results_list, key=lambda x: x['test_accuracy'])
    print(f"\n🏆 Best Advanced Model: {best_model['model_name']} ({best_model['test_accuracy']:.4f})")
    
    # Performance comparison
    print("\n📊 PERFORMANCE RANKING:")
    print(performance_df[['Model', 'Overall Score', 'Test Accuracy', 'ROC-AUC']].head())
    
    return trained_models, results_list, scaler, label_encoder, performance_df

if __name__ == "__main__":
    trained_models, results_list, scaler, label_encoder, performance_df = main()
