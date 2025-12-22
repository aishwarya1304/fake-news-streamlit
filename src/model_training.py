"""
MODEL TRAINING MODULE
====================
Functions for training and evaluating ML models
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import numpy as np

def train_models(models_dict, X_train, y_train):
    """
    Train multiple models
    
    Args:
        models_dict (dict): Dictionary of model names and instances
        X_train: Training features
        y_train: Training labels
        
    Returns:
        dict: Trained models
    """
    trained_models = {}
    
    for name, model in models_dict.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} trained")
    
    return trained_models

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a single model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
        
    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print results
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print('='*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    return metrics

def get_feature_importance(model, vectorizer, top_n=20):
    """
    Get most important features for classification
    
    Args:
        model: Trained model with coef_ attribute
        vectorizer: Fitted TF-IDF vectorizer
        top_n (int): Number of top features to return
        
    Returns:
        tuple: (fake_words, real_words)
    """
    if not hasattr(model, 'coef_'):
        return None, None
    
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    
    # Top fake news indicators (highest positive coefficients)
    top_fake_indices = np.argsort(coef)[-top_n:][::-1]
    top_fake_words = [feature_names[i] for i in top_fake_indices]
    
    # Top real news indicators (highest negative coefficients)
    top_real_indices = np.argsort(coef)[:top_n]
    top_real_words = [feature_names[i] for i in top_real_indices]
    
    return top_fake_words, top_real_words

def calculate_metrics_detailed(y_true, y_pred):
    """
    Calculate detailed metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Detailed metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }
    
    return metrics