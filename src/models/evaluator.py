"""
Model Evaluator Module for Phishing Detection

This module handles the evaluation of machine learning models for phishing detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, classification_report
)
import joblib
import os
import json


class ModelEvaluator:
    """Evaluate phishing detection models and visualize their performance."""
    
    def __init__(self, output_dir='models'):
        """Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model on the test set.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Check if the model is a neural network (TensorFlow model)
        is_tf_model = hasattr(model, 'predict')
        
        # Make predictions
        if is_tf_model:
            y_proba = model.predict(X_test)
            y_pred = (y_proba > 0.5).astype(int)
        elif hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            y_proba = y_pred  # Binary prediction if probabilities not available
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC AUC if probabilities are available
        if hasattr(model, 'predict_proba') or is_tf_model:
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Assemble metrics
        metrics = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Print results
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, f"{model_name.lower().replace(' ', '_')}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot a confusion matrix.
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            model_name (str): Name of the model
            
        Returns:
            matplotlib.figure.Figure: Confusion matrix figure
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure
        cm_path = os.path.join(self.output_dir, f"{model_name.lower().replace(' ', '_')}_cm.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, y_test, y_proba, model_name):
        """Plot ROC curve.
        
        Args:
            y_test: Test target
            y_proba: Predicted probabilities
            model_name (str): Name of the model
            
        Returns:
            matplotlib.figure.Figure: ROC curve figure
        """
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        # Save figure
        roc_path = os.path.join(self.output_dir, f"{model_name.lower().replace(' ', '_')}_roc.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, y_test, y_proba, model_name):
        """Plot precision-recall curve.
        
        Args:
            y_test: Test target
            y_proba: Predicted probabilities
            model_name (str): Name of the model
            
        Returns:
            matplotlib.figure.Figure: Precision-recall curve figure
        """
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        
        # Save figure
        pr_path = os.path.join(self.output_dir, f"{model_name.lower().replace(' ', '_')}_pr.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def compare_models(self, model_metrics_list):
        """Compare multiple models using their evaluation metrics.
        
        Args:
            model_metrics_list (list): List of model metrics dictionaries
            
        Returns:
            pandas.DataFrame: Comparison dataframe
        """
        # Extract key metrics for comparison
        comparison_data = []
        for metrics in model_metrics_list:
            model_data = {
                'Model': metrics['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics['roc_auc'] if 'roc_auc' in metrics else None
            }
            comparison_data.append(model_data)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        # Plot comparison
        self.plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df):
        """Plot model comparison.
        
        Args:
            comparison_df (pandas.DataFrame): Comparison dataframe
            
        Returns:
            matplotlib.figure.Figure: Comparison figure
        """
        # Melt the dataframe for easier plotting
        melted_df = pd.melt(
            comparison_df, 
            id_vars=['Model'], 
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            var_name='Metric', value_name='Value'
        )
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Model', y='Value', hue='Metric', data=melted_df)
        plt.title('Model Comparison')
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save figure
        comparison_path = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def find_best_model(self, model_metrics_list, metric='f1_score'):
        """Find the best model based on a specific metric.
        
        Args:
            model_metrics_list (list): List of model metrics dictionaries
            metric (str): Metric to use for comparison
            
        Returns:
            dict: Best model metrics
        """
        # Validate metric
        valid_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Valid metrics: {valid_metrics}")
        
        # Find best model
        best_model_idx = max(
            range(len(model_metrics_list)),
            key=lambda i: model_metrics_list[i][metric] if model_metrics_list[i][metric] is not None else 0
        )
        best_model = model_metrics_list[best_model_idx]
        
        print(f"\nBest model based on {metric}: {best_model['model_name']}")
        print(f"{metric.replace('_', ' ').title()}: {best_model[metric]:.4f}")
        
        # Save best model info
        best_model_path = os.path.join(self.output_dir, "best_model_info.json")
        with open(best_model_path, 'w') as f:
            json.dump({
                'model_name': best_model['model_name'],
                'metric': metric,
                'value': best_model[metric],
                'all_metrics': best_model
            }, f, indent=4)
        
        return best_model
    
    def evaluate_feature_importance(self, model, feature_names, model_name):
        """Evaluate feature importance for a model.
        
        Args:
            model: Trained model object
            feature_names (list): List of feature names
            model_name (str): Name of the model
            
        Returns:
            dict: Feature importance dictionary
        """
        # Check if model has feature importances
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            print(f"Model {model_name} does not have feature importances")
            return None
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        # Create dictionary of feature importances
        feature_importance = {
            name: float(importance) for name, importance in zip(feature_names, importances)
        }
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Plot feature importances
        self.plot_feature_importance(sorted_features, model_name)
        
        # Save feature importances
        importance_path = os.path.join(
            self.output_dir, 
            f"{model_name.lower().replace(' ', '_')}_feature_importance.json"
        )
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f, indent=4)
        
        return feature_importance
    
    def plot_feature_importance(self, sorted_features, model_name, top_n=20):
        """Plot feature importance.
        
        Args:
            sorted_features (list): List of (feature, importance) tuples
            model_name (str): Name of the model
            top_n (int): Number of top features to plot
            
        Returns:
            matplotlib.figure.Figure: Feature importance figure
        """
        # Get top N features
        top_features = sorted_features[:top_n]
        
        # Plot
        plt.figure(figsize=(12, 8))
        names = [x[0] for x in top_features]
        values = [x[1] for x in top_features]
        
        plt.barh(range(len(names)), values, align='center')
        plt.yticks(range(len(names)), names)
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save figure
        importance_path = os.path.join(
            self.output_dir, 
            f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        )
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


def evaluate_models(models_list, X_test, y_test, feature_names=None):
    """Evaluate multiple models and compare their performance.
    
    Args:
        models_list (list): List of (model, model_name) tuples
        X_test: Test features
        y_test: Test target
        feature_names (list, optional): List of feature names
        
    Returns:
        tuple: (best_model, all_metrics)
    """
    evaluator = ModelEvaluator()
    all_metrics = []
    
    # Evaluate each model
    for model, model_name in models_list:
        print(f"\nEvaluating {model_name}...")
        
        # Calculate metrics
        metrics = evaluator.evaluate_model(model, X_test, y_test, model_name)
        all_metrics.append(metrics)
        
        # Plot confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        evaluator.plot_confusion_matrix(cm, model_name)
        
        # Plot ROC curve if probabilities are available
        if hasattr(model, 'predict_proba') or hasattr(model, 'predict'):
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:  # Neural network
                y_proba = model.predict(X_test).flatten()
            
            evaluator.plot_roc_curve(y_test, y_proba, model_name)
            evaluator.plot_precision_recall_curve(y_test, y_proba, model_name)
        
        # Evaluate feature importance if feature names are provided
        if feature_names is not None:
            evaluator.evaluate_feature_importance(model, feature_names, model_name)
    
    # Compare models
    evaluator.compare_models(all_metrics)
    
    # Find best model
    best_model = evaluator.find_best_model(all_metrics)
    
    return best_model, all_metrics


if __name__ == "__main__":
    # Example usage
    
    # Load models
    # random_forest = joblib.load('models/random_forest.pkl')
    # svm_model = joblib.load('models/svm_model.pkl')
    # neural_network = tf.keras.models.load_model('models/neural_network')
    
    # Load test data
    # X_test = ...
    # y_test = ...
    
    # Feature names
    # feature_names = [...]
    
    # Evaluate models
    # models_list = [
    #     (random_forest, 'Random Forest'),
    #     (svm_model, 'SVM'),
    #     (neural_network, 'Neural Network')
    # ]
    # best_model, all_metrics = evaluate_models(models_list, X_test, y_test, feature_names)
    
    print("Model evaluator module loaded successfully!")