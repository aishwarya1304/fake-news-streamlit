"""
PREDICTION MODULE
================
Functions for loading models and making predictions
"""

import pickle
import os
import numpy as np

def load_saved_model(model_path='models/fake_news_model.pkl'):
    """
    Load trained model from pickle file
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        model or None: Loaded model if successful
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_saved_vectorizer(vectorizer_path='models/tfidf_vectorizer.pkl'):
    """
    Load trained vectorizer from pickle file
    
    Args:
        vectorizer_path (str): Path to vectorizer file
        
    Returns:
        vectorizer or None: Loaded vectorizer if successful
    """
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✅ Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    except FileNotFoundError:
        print(f"❌ Vectorizer file not found: {vectorizer_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading vectorizer: {e}")
        return None

def predict_news(text, model, vectorizer):
    """
    Predict if news is fake or real
    
    Args:
        text (str): News text to predict
        model: Trained ML model
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        tuple: (prediction, confidence)
            prediction: 1 for fake, 0 for real
            confidence: confidence score (0-100)
    """
    try:
        from preprocessing import preprocess_text
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        
        # Get confidence/probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None
        
        return prediction, confidence
    
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return 1, None  # Default to fake if error