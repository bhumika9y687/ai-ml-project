# ====================
# 1. IMPORT LIBRARIES
# ====================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, precision_recall_curve, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier

# Advanced Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Handling Imbalanced Data
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier

# Radar chart
from math import pi

# Additional utilities
import joblib
import json
import os
from datetime import datetime
import sys
import traceback

# ====================
# 2. CONFIGURATION
# ====================
TARGET_PERFORMANCE = 0.97
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ====================
# 3. LOAD AND ENHANCE DATA
# ====================
def load_and_enhance_data(filepath='data/kaggle_heart.csv'):
    """Load dataset and create enhanced features"""
    print("=" * 60)
    print("🚀 LOADING AND ENHANCING DATA")
    print("=" * 60)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Successfully loaded data from {filepath}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {filepath}")
        print("Creating sample dataset for testing...")
        np.random.seed(RANDOM_STATE)
        n_samples = 1000
        df = pd.DataFrame({
            'HeartDiseaseorAttack': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'HighBP': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'HighChol': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'CholCheck': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'BMI': np.random.normal(28, 6, n_samples).clip(18, 50),
            'Smoker': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'Stroke': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'Diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'PhysActivity': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'Fruits': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Veggies': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'HvyAlcoholConsump': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'AnyHealthcare': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'NoDocbcCost': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'GenHlth': np.random.randint(1, 6, n_samples),
            'MentHlth': np.random.randint(0, 31, n_samples),
            'PhysHlth': np.random.randint(0, 31, n_samples),
            'DiffWalk': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Sex': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'Age': np.random.randint(1, 14, n_samples),
            'Education': np.random.randint(1, 7, n_samples),
            'Income': np.random.randint(1, 9, n_samples)
        })
    
    original_shape = df.shape
    
    # Create interaction features
    print("Creating interaction features...")
    
    # BMI interactions
    if 'BMI' in df.columns and 'GenHlth' in df.columns:
        df['BMI_GenHlth'] = df['BMI'] * df['GenHlth']
    if 'BMI' in df.columns and 'Age' in df.columns:
        df['BMI_Age'] = df['BMI'] * df['Age']
    if 'BMI' in df.columns and 'HighBP' in df.columns:
        df['BMI_HighBP'] = df['BMI'] * df['HighBP']
    if 'BMI' in df.columns and 'Smoker' in df.columns:
        df['BMI_Smoker'] = df['BMI'] * df['Smoker']
    
    # Age interactions
    if 'Age' in df.columns and 'HighBP' in df.columns:
        df['Age_HighBP'] = df['Age'] * df['HighBP']
    if 'Age' in df.columns and 'HighChol' in df.columns:
        df['Age_HighChol'] = df['Age'] * df['HighChol']
    if 'Age' in df.columns and 'Smoker' in df.columns:
        df['Age_Smoker'] = df['Age'] * df['Smoker']
    
    # Risk score features
    if all(col in df.columns for col in ['HighBP', 'HighChol', 'Smoker', 'Diabetes', 'BMI', 'GenHlth']):
        df['Risk_Score'] = (df['HighBP'] * 2 + df['HighChol'] * 1.5 + 
                           df['Smoker'] * 2 + df['Diabetes'] * 2.5)
        df['Metabolic_Score'] = (df['HighBP'] + df['HighChol'] + 
                                df['BMI']/30 + df['GenHlth']/5)
    
    # Polynomial features
    if 'BMI' in df.columns:
        df['BMI_squared'] = df['BMI'] ** 2
    if 'Age' in df.columns:
        df['Age_squared'] = df['Age'] ** 2
    
    # Create health behavior composite
    health_cols = ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump']
    if all(col in df.columns for col in health_cols):
        df['Healthy_Lifestyle'] = df[health_cols].sum(axis=1)
    
    print(f"Original shape: {original_shape}")
    print(f"Enhanced shape: {df.shape}")
    print(f"Added {df.shape[1] - original_shape[1]} new features")
    
    if 'HeartDiseaseorAttack' not in df.columns:
        print("❌ ERROR: Target column 'HeartDiseaseorAttack' not found!")
        sys.exit(1)
    
    return df

# ====================
# 4. DATA PREPROCESSING
# ====================
def advanced_preprocessing(df):
    """Preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("🔧 DATA PREPROCESSING")
    print("=" * 60)
    
    if 'HeartDiseaseorAttack' not in df.columns:
        print("❌ ERROR: 'HeartDiseaseorAttack' column not found!")
        return None, None, None, None, None, None, None, None
    
    X = df.drop('HeartDiseaseorAttack', axis=1)
    y = df['HeartDiseaseorAttack'].astype(int)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Remove extreme outliers
    print("1. Handling outliers...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        Q1 = X[numeric_cols].quantile(0.25)
        Q3 = X[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outlier_mask = ((X[numeric_cols] >= lower_bound) & (X[numeric_cols] <= upper_bound)).all(axis=1)
        X = X[outlier_mask]
        y = y[outlier_mask]
        print(f"   Removed {len(outlier_mask) - sum(outlier_mask)} outliers")
    
    # Scaling
    print("2. Applying scaling...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Oversampling
    print("3. Applying oversampling...")
    print(f"   Before - Class 0: {(y == 0).sum():,}, Class 1: {(y == 1).sum():,}")
    
    try:
        borderline = BorderlineSMOTE(random_state=RANDOM_STATE, sampling_strategy=0.9, kind='borderline-2')
        X_resampled, y_resampled = borderline.fit_resample(X_scaled, y)
        adasyn = ADASYN(random_state=RANDOM_STATE, n_neighbors=3)
        X_resampled, y_resampled = adasyn.fit_resample(X_resampled, y_resampled)
        print(f"   After - Class 0: {(y_resampled == 0).sum():,}, Class 1: {(y_resampled == 1).sum():,}")
    except Exception as e:
        print(f"   Warning: {e}. Using original data.")
        X_resampled, y_resampled = X_scaled, y
    
    # Train-test split
    print("4. Creating train-test split...")
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_resampled, y_resampled, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_resampled
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=RANDOM_STATE, stratify=y_temp
        )
    except Exception as e:
        print(f"   Error: {e}. Using simple split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_val, y_val = X_test, y_test
    
    print(f"\nFinal dataset sizes:")
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, X.columns

# ====================
# 5. VISUALIZATION FUNCTIONS
# ====================
def create_confusion_matrices(models, X_test, y_test):
    """Create confusion matrices for all models"""
    print("\n" + "=" * 60)
    print("📊 CREATING CONFUSION MATRICES")
    print("=" * 60)
    
    os.makedirs('visualizations/confusion_matrices', exist_ok=True)
    
    # Create individual confusion matrices for each model
    for model_name, model in models.items():
        if model is not None:
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No Disease', 'Disease'],
                           yticklabels=['No Disease', 'Disease'])
                plt.title(f'Confusion Matrix - {model_name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                # Add metrics text
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
                plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
                
                # Save figure
                filename = f'visualizations/confusion_matrices/confusion_matrix_{model_name.replace(" ", "_")}.png'
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Created confusion matrix for {model_name}")
                
            except Exception as e:
                print(f"❌ Error creating confusion matrix for {model_name}: {e}")
    
    # Create grid of confusion matrices for comparison
    create_confusion_matrices_grid(models, X_test, y_test)
    
    # Create enhanced confusion matrices with percentages
    create_enhanced_confusion_matrices(models, X_test, y_test)

def create_confusion_matrices_grid(models, X_test, y_test):
    """Create a grid of confusion matrices for all models"""
    print("\nCreating confusion matrices grid...")
    
    # Filter out None models
    valid_models = {name: model for name, model in models.items() if model is not None}
    
    if not valid_models:
        print("❌ No valid models for confusion matrices grid")
        return
    
    # Determine grid size
    n_models = len(valid_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Flatten axes if needed
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (model_name, model) in enumerate(valid_models.items()):
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'])
            ax.set_title(f'{model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
        except Exception as e:
            print(f"❌ Error in grid for {model_name}: {e}")
    
    # Hide unused subplots
    for idx in range(len(valid_models), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created confusion matrices grid")

def create_enhanced_confusion_matrices(models, X_test, y_test):
    """Create enhanced confusion matrices with percentages"""
    print("\nCreating enhanced confusion matrices with percentages...")
    
    for model_name, model in models.items():
        if model is not None:
            try:
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                # Calculate percentages
                cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot absolute values
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                           xticklabels=['No Disease', 'Disease'],
                           yticklabels=['No Disease', 'Disease'])
                ax1.set_title(f'{model_name} - Absolute Values')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Actual')
                
                # Plot percentages
                sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                           xticklabels=['No Disease', 'Disease'],
                           yticklabels=['No Disease', 'Disease'])
                ax2.set_title(f'{model_name} - Percentage (%)')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Actual')
                
                # Calculate and display metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
                plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
                
                plt.suptitle(f'Enhanced Analysis - {model_name}', fontsize=14)
                plt.tight_layout()
                
                filename = f'visualizations/confusion_matrices/enhanced_confusion_matrix_{model_name.replace(" ", "_")}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Created enhanced confusion matrix for {model_name}")
                
            except Exception as e:
                print(f"❌ Error creating enhanced confusion matrix for {model_name}: {e}")

def create_radar_comparison(results):
    """Create radar chart for model comparison"""
    print("\n" + "=" * 60)
    print("📡 CREATING RADAR CHART COMPARISON")
    print("=" * 60)
    
    os.makedirs('visualizations/radar_charts', exist_ok=True)
    
    if not results:
        print("❌ No results for radar chart")
        return
    
    # Select metrics for radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
    
    # Prepare data
    model_names = []
    model_metrics = []
    
    for model_name, metrics_dict in results.items():
        if metrics_dict:
            model_names.append(model_name)
            model_values = []
            for metric in metrics:
                value = metrics_dict.get(metric, 0)
                # Normalize to 0-1 scale
                model_values.append(value)
            model_metrics.append(model_values)
    
    if not model_metrics:
        print("❌ No valid metrics for radar chart")
        return
    
    # Number of variables
    N = len(metrics)
    
    # Create radar chart
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Calculate angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set up the plot
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], metrics)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Colors for different models
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    # Plot each model
    for idx, (model_name, values) in enumerate(zip(model_names, model_metrics)):
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Model Comparison - Radar Chart', size=16, y=1.1)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('visualizations/radar_charts/model_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created radar chart comparison")
    
    # Create individual radar charts for each model
    create_individual_radar_charts(results)

def create_individual_radar_charts(results):
    """Create individual radar charts for each model"""
    print("\nCreating individual radar charts...")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
    N = len(metrics)
    
    for model_name, metrics_dict in results.items():
        if metrics_dict:
            try:
                values = [metrics_dict.get(metric, 0) for metric in metrics]
                
                # Create radar chart
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, polar=True)
                
                angles = [n / float(N) * 2 * pi for n in range(N)]
                angles += angles[:1]
                values += values[:1]
                
                ax.set_theta_offset(pi / 2)
                ax.set_theta_direction(-1)
                plt.xticks(angles[:-1], metrics)
                
                ax.set_rlabel_position(0)
                plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
                plt.ylim(0, 1)
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', color='steelblue')
                ax.fill(angles, values, alpha=0.25, color='steelblue')
                
                # Add value labels
                for angle, value in zip(angles[:-1], values[:-1]):
                    ax.text(angle, value + 0.05, f'{value:.3f}', 
                           ha='center', va='center', fontsize=9)
                
                plt.title(f'Performance Metrics - {model_name}', size=14, y=1.1)
                
                # Add average score
                avg_score = np.mean(values[:-1])
                plt.figtext(0.5, 0.02, f'Average Score: {avg_score:.3f}', 
                           ha='center', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
                
                plt.tight_layout()
                filename = f'visualizations/radar_charts/radar_{model_name.replace(" ", "_")}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Created individual radar chart for {model_name}")
                
            except Exception as e:
                print(f"❌ Error creating radar chart for {model_name}: {e}")

def create_comprehensive_visualizations(models, results, X_test, y_test, feature_names, scaler):
    """Create all visualizations"""
    print("\n" + "=" * 60)
    print("🎨 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Confusion matrices
    create_confusion_matrices(models, X_test, y_test)
    
    # 2. Radar charts
    create_radar_comparison(results)
    
    # 3. ROC Curves
    create_roc_curves(models, X_test, y_test)
    
    # 4. Performance comparison bar charts
    create_performance_comparison(results)
    
    # 5. Feature importance for tree-based models
    create_feature_importance(models, scaler, feature_names)

def create_roc_curves(models, X_test, y_test):
    """Create ROC curves for all models"""
    print("\nCreating ROC curves...")
    
    os.makedirs('visualizations/roc_curves', exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for idx, (model_name, model) in enumerate(models.items()):
        if model is not None:
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)
                plt.plot(fpr, tpr, lw=2, color=colors[idx], 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"   Skipping {model_name}: {e}")
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig('visualizations/roc_curves_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created ROC curves")

def create_performance_comparison(results):
    """Create performance comparison bar charts"""
    print("\nCreating performance comparison charts...")
    
    os.makedirs('visualizations/performance_comparison', exist_ok=True)
    
    if not results:
        return
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        model_names = []
        metric_values = []
        
        for model_name, metrics_dict in results.items():
            if metrics_dict and metric in metrics_dict:
                model_names.append(model_name)
                metric_values.append(metrics_dict[metric])
        
        if model_names:
            bars = plt.bar(model_names, metric_values, color=plt.cm.Set3(range(len(model_names))))
            plt.xlabel('Models')
            plt.ylabel(metric)
            plt.title(f'{metric} Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'visualizations/performance_comparison/{metric.lower()}_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print("✅ Created performance comparison charts")

def create_feature_importance(models, scaler, feature_names):
    """Create feature importance plots for tree-based models"""
    print("\nCreating feature importance plots...")
    
    os.makedirs('visualizations/feature_importance', exist_ok=True)
    
    tree_models = {}
    for model_name, model in models.items():
        if model is not None and hasattr(model, 'feature_importances_'):
            tree_models[model_name] = model
    
    if tree_models:
        for model_name, model in tree_models.items():
            try:
                importances = model.feature_importances_
                indices = np.argsort(importances)[-15:]  # Top 15 features
                
                plt.figure(figsize=(10, 8))
                
                # Use original feature names if available
                if hasattr(scaler, 'feature_names_in_'):
                    feature_names_list = list(scaler.feature_names_in_)
                else:
                    feature_names_list = list(feature_names) if feature_names is not None else [f'Feature {i}' for i in range(len(importances))]
                
                plt.barh(range(len(indices)), importances[indices], color='skyblue')
                plt.yticks(range(len(indices)), [feature_names_list[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title(f'{model_name} - Top 15 Feature Importances')
                plt.tight_layout()
                
                plt.savefig(f'visualizations/feature_importance/feature_importance_{model_name.replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Created feature importance plot for {model_name}")
                
            except Exception as e:
                print(f"❌ Error creating feature importance for {model_name}: {e}")

# ====================
# 6. MODEL BUILDING WITH ALL MODELS
# ====================
def build_all_models(X_train, X_val, X_test, y_train, y_val, y_test, scaler):
    """Build ALL models and combine them"""
    
    models = {}
    results = {}
    
    if X_train is None or len(X_train) == 0:
        print("❌ ERROR: No training data!")
        return models, results
    
    # ========== 1. EXTREME RANDOM FOREST ==========
    print("\n" + "="*60)
    print("🌲 1. EXTREME RANDOM FOREST")
    print("="*60)
    
    try:
        rf_model = BalancedRandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            replacement=True,
            sampling_strategy='all',
            n_jobs=-1,
            verbose=0
        )
        
        param_grid = {
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        print(f"✅ Trained. Best params: {grid_search.best_params_}")
        
        rf_metrics = evaluate_model(best_rf, "Extreme_Random_Forest", X_train, X_val, X_test, y_train, y_val, y_test)
        models['Extreme_Random_Forest'] = best_rf
        results['Extreme_Random_Forest'] = rf_metrics
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ========== 2. XGBOOST ==========
    print("\n" + "="*60)
    print("⚡ 2. XGBOOST")
    print("="*60)
    
    try:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        
        xgb_model = XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1,
            verbosity=0
        )
        
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("✅ Trained")
        
        xgb_metrics = evaluate_model(xgb_model, "XGBoost", X_train, X_val, X_test, y_train, y_val, y_test)
        models['XGBoost'] = xgb_model
        results['XGBoost'] = xgb_metrics
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ========== 3. LIGHTGBM ==========
    print("\n" + "="*60)
    print("💡 3. LIGHTGBM")
    print("="*60)
    
    try:
        lgb_model = LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("✅ Trained")
        
        lgb_metrics = evaluate_model(lgb_model, "LightGBM", X_train, X_val, X_test, y_train, y_val, y_test)
        models['LightGBM'] = lgb_model
        results['LightGBM'] = lgb_metrics
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ========== 4. CATBOOST ==========
    print("\n" + "="*60)
    print("🐱 4. CATBOOST")
    print("="*60)
    
    try:
        cat_model = CatBoostClassifier(
            random_state=RANDOM_STATE,
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            border_count=128,
            loss_function='Logloss',
            verbose=False,
            auto_class_weights='Balanced'
        )
        
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        print("✅ Trained")
        
        cat_metrics = evaluate_model(cat_model, "CatBoost", X_train, X_val, X_test, y_train, y_val, y_test)
        models['CatBoost'] = cat_model
        results['CatBoost'] = cat_metrics
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ========== 5. NEURAL NETWORK ==========
    print("\n" + "="*60)
    print("🧠 5. NEURAL NETWORK")
    print("="*60)
    
    try:
        nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False
        )
        
        nn_model.fit(X_train, y_train)
        print("✅ Trained")
        
        nn_metrics = evaluate_model(nn_model, "Neural_Network", X_train, X_val, X_test, y_train, y_val, y_test)
        models['Neural_Network'] = nn_model
        results['Neural_Network'] = nn_metrics
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return models, results

# ====================
# 7. EVALUATION FUNCTION
# ====================
def evaluate_model(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate a single model"""
    
    print(f"\n📊 Evaluating {model_name}")
    print("-" * 40)
    
    # Get probabilities
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # Specificity
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0
    
    # Balanced accuracy
    balanced_acc = (recall + specificity) / 2
    
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print(f"ROC-AUC:        {roc_auc:.4f}")
    print(f"Specificity:    {specificity:.4f}")
    print(f"Balanced Acc:   {balanced_acc:.4f}")
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Specificity': specificity,
        'Balanced_Accuracy': balanced_acc
    }

# ====================
# 8. SAVE MODELS AND RESULTS
# ====================
def save_models_and_results(models, scaler, results, comparison_df):
    """Save models and results separately"""
    
    print("\n" + "="*60)
    print("💾 SAVING MODELS AND RESULTS")
    print("="*60)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. Save each individual model
    print("\n💽 Saving individual models:")
    for model_name, model in models.items():
        if model is not None:
            try:
                filename = f'models/{model_name}.pkl'
                joblib.dump(model, filename)
                print(f"✅ {model_name:25s} → {filename}")
            except Exception as e:
                print(f"❌ Failed to save {model_name}: {e}")
    
    # 2. Save the scaler
    try:
        joblib.dump(scaler, 'models/scaler.pkl')
        print(f"\n✅ Saved scaler: models/scaler.pkl")
    except Exception as e:
        print(f"❌ Failed to save scaler: {e}")
    
    # 3. Save results to JSON
    try:
        results_serializable = {}
        for model_name, metrics in results.items():
            if metrics:
                results_serializable[model_name] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v
                    for k, v in metrics.items()
                }
        
        with open('results/all_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=4)
        print(f"\n✅ Saved results: results/all_results.json")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # 4. Save comparison CSV
    if comparison_df is not None and not comparison_df.empty:
        try:
            comparison_df.to_csv('results/model_comparison.csv', index=False)
            print(f"✅ Saved comparison: results/model_comparison.csv")
        except Exception as e:
            print(f"❌ Failed to save comparison: {e}")
    
    # 5. Save detailed classification reports
    save_detailed_reports(models, results)

def save_detailed_reports(models, results):
    """Save detailed classification reports for each model"""
    print("\n📝 Saving detailed classification reports...")
    
    detailed_reports = {}
    
    for model_name, model in models.items():
        if model is not None and model_name in results:
            try:
                # Create a comprehensive report
                report = {
                    'metrics': results[model_name],
                    'model_type': str(type(model)).split('.')[-1].replace("'>", ""),
                    'model_params': model.get_params() if hasattr(model, 'get_params') else {}
                }
                detailed_reports[model_name] = report
            except Exception as e:
                print(f"❌ Error creating report for {model_name}: {e}")
    
    # Save to JSON
    try:
        with open('results/detailed_reports.json', 'w') as f:
            json.dump(detailed_reports, f, indent=4, default=str)
        print(f"✅ Saved detailed reports: results/detailed_reports.json")
    except Exception as e:
        print(f"❌ Failed to save detailed reports: {e}")

# ====================
# 9. MAIN EXECUTION
# ====================
def main():
    """Main pipeline"""
    
    print("\n" + "="*70)
    print("🚀 HEART DISEASE PREDICTION - COMPREHENSIVE PIPELINE")
    print("="*70)
    print(f"🎯 Training models with confusion matrices and radar charts")
    print("="*70)
    
    try:
        # Create directories
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # 1. Load and enhance data
        df = load_and_enhance_data()
        
        if df is None or len(df) == 0:
            print("❌ ERROR: No data loaded!")
            return
        
        # 2. Preprocessing
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = advanced_preprocessing(df)
        
        if X_train is None:
            print("❌ ERROR: Preprocessing failed!")
            return
        
        # 3. Build ALL models
        models, results = build_all_models(X_train, X_val, X_test, y_train, y_val, y_test, scaler)
        
        # 4. Create comparison dataframe
        if results:
            comparison_data = []
            for model_name, metrics in results.items():
                if metrics:
                    comparison_data.append({
                        'Model': model_name,
                        'Accuracy': metrics.get('Accuracy', 0),
                        'Precision': metrics.get('Precision', 0),
                        'Recall': metrics.get('Recall', 0),
                        'F1-Score': metrics.get('F1-Score', 0),
                        'ROC-AUC': metrics.get('ROC-AUC', 0),
                        'Specificity': metrics.get('Specificity', 0),
                        'Balanced_Accuracy': metrics.get('Balanced_Accuracy', 0)
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
            
            print("\n" + "="*60)
            print("🏆 MODEL COMPARISON (Sorted by F1-Score)")
            print("="*60)
            print(comparison_df.to_string(index=False))
        else:
            comparison_df = pd.DataFrame()
            print("❌ No results to compare")
        
        # 5. Create comprehensive visualizations
        create_comprehensive_visualizations(models, results, X_test, y_test, feature_names, scaler)
        
        # 6. Save models and results separately
        save_models_and_results(models, scaler, results, comparison_df)
        
        # 7. Final report
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        
        if comparison_df is not None and not comparison_df.empty:
            best_model = comparison_df.iloc[0]['Model']
            best_f1 = comparison_df.iloc[0]['F1-Score']
            print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.2%})")
        
        print("\n📁 OUTPUTS CREATED:")
        print("   - models/ (individual .pkl model files)")
        print("   - results/ (JSON results and CSV comparison)")
        print("   - visualizations/ (comprehensive visualizations)")
        print("\n📊 VISUALIZATIONS INCLUDE:")
        print("   - Individual confusion matrices for each model")
        print("   - Confusion matrices grid comparison")
        print("   - Enhanced confusion matrices with percentages")
        print("   - Radar charts for model comparison")
        print("   - Individual radar charts for each model")
        print("   - ROC curves for all models")
        print("   - Performance comparison bar charts")
        print("   - Feature importance plots for tree-based models")
        
    except Exception as e:
        print(f"\n❌ ERROR in main execution: {e}")
        traceback.print_exc()

# ====================
# 10. RUN PIPELINE
# ====================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"🚀 Starting comprehensive pipeline at {start_time}")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n⏱️  Total execution time: {duration}")