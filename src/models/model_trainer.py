"""
Model Trainer for Phishing Detection

This module handles the training and evaluation of machine learning models
for phishing email detection.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class PhishingModelTrainer:
    """Train and evaluate models for phishing detection."""
    
    def __init__(self, models_dir='models'):
        """Initialize the model trainer.
        
        Args:
            models_dir (str): Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.scaler = StandardScaler()
        
    def prepare_data(self, features_df, target_column='is_phishing', test_size=0.2, val_size=0.25):
        """Prepare data for model training.
        
        Args:
            features_df (pandas.DataFrame): DataFrame with features and target
            target_column (str): Name of the target column
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Separate features and target
        X = features_df.drop(columns=[target_column])
        y = features_df[target_column]
        
        # Split into train and test sets
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Split training set into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=42, stratify=y_train_full
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train a Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained model
        """
        print("Training Random Forest model...")
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create grid search with cross-validation
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_rf = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = best_rf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        print(f"Random Forest - Best parameters: {grid_search.best_params_}")
        print(f"Random Forest - Validation accuracy: {val_accuracy:.4f}")
        print(f"Random Forest - Validation F1: {val_f1:.4f}")
        
        # Save the model
        joblib.dump(best_rf, os.path.join(self.models_dir, 'random_forest.pkl'))
        
        return best_rf
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """Train an SVM classifier.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            sklearn.svm.SVC: Trained model
        """
        print("Training SVM model...")
        
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
        
        # Create grid search with cross-validation
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_svm = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = best_svm.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        print(f"SVM - Best parameters: {grid_search.best_params_}")
        print(f"SVM - Validation accuracy: {val_accuracy:.4f}")
        print(f"SVM - Validation F1: {val_f1:.4f}")
        
        # Save the model
        joblib.dump(best_svm, os.path.join(self.models_dir, 'svm_model.pkl'))
        
        return best_svm
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train a neural network classifier.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            tensorflow.keras.models.Sequential: Trained model
        """
        print("Training Neural Network model...")
        
        # Convert to TensorFlow format
        X_val_tf = np.array(X_val)
        y_val_tf = np.array(y_val)
        
        # Get input shape
        input_dim = X_train.shape[1]
        
        # Create a simple neural network
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_tf, y_val_tf),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate on validation set
        y_val_pred_proba = model.predict(X_val_tf)
        y_val_pred = (y_val_pred_proba > 0.5).astype(int)
        val_accuracy = accuracy_score(y_val_tf, y_val_pred)
        val_f1 = f1_score(y_val_tf, y_val_pred)
        
        print(f"Neural Network - Validation accuracy: {val_accuracy:.4f}")
        print(f"Neural Network - Validation F1: {val_f1:.4f}")
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'neural_network')
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model on the test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:  # Neural network
            y_pred_proba = model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"{model_name} - Test accuracy: {accuracy:.4f}")
        print(f"{model_name} - Test precision: {precision:.4f}")
        print(f"{model_name} - Test recall: {recall:.4f}")
        print(f"{model_name} - Test F1: {f1:.4f}")
        print(f"{model_name} - Test AUC-ROC: {auc_roc:.4f}")
        print(f"{model_name} - Confusion Matrix:\n{cm}")
        print(f"{model_name} - Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Return metrics
        return {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm
        }
    
    def train_all_models(self, features_df, target_column='is_phishing'):
        """Train all models and select the best one.
        
        Args:
            features_df (pandas.DataFrame): DataFrame with features and target
            target_column (str): Name of the target column
            
        Returns:
            dict: Best model and its evaluation metrics
        """
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            features_df, target_column
        )
        
        # Train all models
        rf_model = self.train_random_forest(X_train, y_train, X_val, y_val)
        svm_model = self.train_svm(X_train, y_train, X_val, y_val)
        nn_model = self.train_neural_network(X_train, y_train, X_val, y_val)
        
        # Evaluate all models
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        svm_metrics = self.evaluate_model(svm_model, X_test, y_test, 'SVM')
        nn_metrics = self.evaluate_model(nn_model, X_test, y_test, 'Neural Network')
        
        # Compare models and select the best one
        models_metrics = [rf_metrics, svm_metrics, nn_metrics]
        best_model_idx = max(range(len(models_metrics)), key=lambda i: models_metrics[i]['f1'])
        best_model_metrics = models_metrics[best_model_idx]
        
        print(f"Best model: {best_model_metrics['model']} with F1 score: {best_model_metrics['f1']:.4f}")
        
        # Return best model info
        return {
            'model_name': best_model_metrics['model'],
            'metrics': best_model_metrics,
            'all_metrics': models_metrics
        }


def tune_model_hyperparameters(X_train, y_train, model_type='random_forest'):
    """Tune hyperparameters for a specific model type.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type (str): Type of model ('random_forest', 'svm', or 'neural_network')
        
    Returns:
        dict: Best hyperparameters
    """
    if model_type == 'random_forest':
        # Define hyperparameter grid for Random Forest
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    elif model_type == 'svm':
        # Define hyperparameter grid for SVM
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'degree': [2, 3, 4]  # Only for poly kernel
        }
    elif model_type == 'logistic_regression':
        # Define hyperparameter grid for Logistic Regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': [None, 'balanced']
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Create grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, {'model__' + k: v for k, v in param_grid.items()},
        cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    # Train with grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    
    print(f"Best parameters for {model_type}: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_params


if __name__ == "__main__":
    # Example usage
    # Load feature data (in a real scenario, this would come from feature extraction)
    features_df = pd.read_csv('../../data/processed/email_features.csv')
    
    # Train models
    trainer = PhishingModelTrainer()
    best_model_info = trainer.train_all_models(features_df)
    
    print(f"Best model: {best_model_info['model_name']}")
    print(f"Accuracy: {best_model_info['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {best_model_info['metrics']['f1']:.4f}")
    print(f"AUC-ROC: {best_model_info['metrics']['auc_roc']:.4f}")