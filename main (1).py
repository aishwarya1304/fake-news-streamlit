# ...existing code...
"""
FAKE NEWS DETECTION - MAIN TRAINING SCRIPT
==========================================
Run this file to train the model

Usage:
    python main.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Try importing helper modules from src; provide safe fallbacks if missing
try:
    from preprocessing import preprocess_text, download_nltk_data
except Exception:
    # Fallback simple text cleaner and no-op downloader
    def download_nltk_data():
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass

    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

try:
    from prediction import predict_news
except Exception:
    # Fallback predict function that works with sklearn models and vectorizers
    import numpy as _np
    def predict_news(text, model, vectorizer):
        txt = preprocess_text(text)
        X = vectorizer.transform([txt])
        try:
            probs = model.predict_proba(X)[0]
            pred = int(probs.argmax())
            confidence = float(probs.max() * 100.0)
            return pred, confidence
        except Exception:
            try:
                scores = model.decision_function(X)
                if hasattr(scores, 'shape') and scores.shape[-1] > 1:
                    # multiclass
                    probs = _np.exp(scores) / _np.exp(scores).sum(axis=1, keepdims=True)
                    pred = int(probs.argmax())
                    confidence = float(probs.max() * 100.0)
                else:
                    score = float(scores[0]) if hasattr(scores, '__iter__') else float(scores)
                    pred = int(model.predict(X)[0])
                    confidence = None
                return pred, confidence
            except Exception:
                pred = int(model.predict(X)[0])
                return pred, None

# Remove unused imports and references to missing module 'model_training' in original file
# ...existing code...
class Config:
    """Project configuration"""
    DATA_PATH = 'data'
    MODEL_PATH = 'models'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)

# ========================================
# SETUP
# ========================================
def setup_directories():
    """Create necessary directories"""
    os.makedirs(Config.DATA_PATH, exist_ok=True)
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    print("✅ Directories created/verified")

def setup_nltk():
    """Download required NLTK data"""
    print("\n📦 Setting up NLTK data...")
    download_nltk_data()
    print("✅ NLTK data ready")

# ========================================
# DATA LOADING
# ========================================
def load_data():
    """Load dataset"""
    print("\n📂 Loading dataset...")
    
    # Try to load from CSV
    csv_files = [
        os.path.join(Config.DATA_PATH, 'fake_news.csv'),
        os.path.join(Config.DATA_PATH, 'news.csv'),
        'fake_news.csv',
        'news.csv'
    ]
    
    for file_path in csv_files:
        if os.path.exists(file_path):
            print(f"✅ Found dataset: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Dataset shape: {df.shape}")
            return df
    
    # If no CSV found, create sample dataset
    print("⚠️ No dataset found. Creating sample dataset...")
    data = create_sample_dataset()
    return data

def create_sample_dataset():
    """Create sample dataset for testing"""
    sample_texts = [
        'Scientists discover breakthrough in cancer treatment research published in peer-reviewed journal',
        'SHOCKING: Celebrity caught in scandal you wont believe what happened click here now',
        'Government announces new economic policy to boost infrastructure development',
        'BREAKING: Aliens landed in downtown officials hiding truth from public urgent',
        'Study shows benefits of regular exercise on mental health and wellbeing',
        'Miracle cure discovered doctors hate this one simple trick lose weight fast',
        'Local school receives federal funding for new technology education program',
        'URGENT: Share this message or face bad luck for ten years forward now',
        'Research team publishes comprehensive findings in nature scientific journal',
        'Click here for FREE money limited time offer act now before its gone',
        'University announces new scholarship program for undergraduate students nationwide',
        'You wont believe this secret that celebrities dont want you to know shocking',
        'Federal Reserve maintains interest rates citing economic stability concerns ongoing',
        'AMAZING: This weird trick will change your life forever click to discover',
        'Climate scientists release new report on global warming trends and predictions',
        'Doctors shocked by this one simple method share before its banned forever',
        'Supreme Court issues ruling on landmark case affecting privacy rights nationwide',
        'BREAKING NEWS: Shocking scandal rocks political establishment share immediately urgent',
        'National health institute releases guidelines for preventive care and wellness',
        'Incredible discovery reveals truth about ancient civilization archaeologists stunned amazed',
    ]
    sample_labels = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    # Expand to a reasonable size
    texts = sample_texts * 50
    labels = sample_labels * 50
    df = pd.DataFrame({'text': texts, 'label': labels})
    df = df.sample(frac=1, random_state=Config.RANDOM_STATE).reset_index(drop=True)
    print(f"✅ Sample dataset created with {len(df)} articles")
    return df

# ========================================
# DATA EXPLORATION
# ========================================
def explore_data(df):
    """Explore and display dataset information"""
    print("\n" + "="*60)
    print("📊 DATA EXPLORATION")
    print("="*60)
    
    print(f"\n📏 Dataset Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    print("\n🔍 First few rows:")
    print(df.head(3))
    
    print("\n❓ Missing Values:")
    print(df.isnull().sum())
    
    print("\n📊 Class Distribution:")
    if 'label' in df.columns:
        print(df['label'].value_counts())
        print(f"Real News (0): {(df['label']==0).sum()}")
        print(f"Fake News (1): {(df['label']==1).sum()}")
    
    return df

# ========================================
# DATA PREPROCESSING
# ========================================
def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n" + "="*60)
    print("🧹 DATA PREPROCESSING")
    print("="*60)
    
    # Ensure required columns exist
    if 'text' not in df.columns:
        if 'title' in df.columns:
            df['text'] = df['title']
        else:
            raise ValueError("No 'text' or 'title' column found in dataset")
    
    if 'label' not in df.columns:
        raise ValueError("No 'label' column found in dataset")
    
    # Remove missing values
    initial_size = len(df)
    df = df.dropna(subset=['text', 'label'])
    print(f"✅ Removed {initial_size - len(df)} rows with missing values")
    
    # Clean text
    print("🔄 Cleaning text data...")
    df['cleaned_text'] = df['text'].astype(str).apply(preprocess_text)
    
    # Show example
    if len(df) > 0:
        print("\n📝 Preprocessing Example:")
        print(f"Before: {df['text'].iloc[0][:100]}...")
        print(f"After:  {df['cleaned_text'].iloc[0][:100]}...")
    
    return df

# ========================================
# FEATURE EXTRACTION
# ========================================
def extract_features(df):
    """Extract TF-IDF features"""
    print("\n" + "="*60)
    print("🔢 FEATURE EXTRACTION")
    print("="*60)
    
    print("⚙️ Applying TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(
        max_features=Config.MAX_FEATURES,
        ngram_range=Config.NGRAM_RANGE,
        min_df=2,
        max_df=0.8
    )
    
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['label'].astype(int)
    
    print(f"✅ TF-IDF Matrix Shape: {X.shape}")
    print(f"✅ Number of Features: {X.shape[1]}")
    print(f"✅ Number of Samples: {X.shape[0]}")
    
    return X, y, tfidf

# ========================================
# TRAIN-TEST SPLIT
# ========================================
def split_data(X, y):
    """Split data into train and test sets"""
    print("\n" + "="*60)
    print("✂️ TRAIN-TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"📚 Training Set: {X_train.shape[0]} samples")
    print(f"📝 Testing Set: {X_test.shape[0]} samples")
    print(f"📊 Split Ratio: {int((1-Config.TEST_SIZE)*100)}% / {int(Config.TEST_SIZE*100)}%")
    
    return X_train, X_test, y_train, y_test

# ========================================
# MODEL TRAINING
# ========================================
def train_all_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare"""
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🔄 Training {name}...")
        print('='*60)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"\n✅ {name} trained successfully!")
        print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        try:
            print(f"\n📋 Classification Report:")
            print(classification_report(y_test, y_pred, 
                                        target_names=['Real News', 'Fake News'],
                                        digits=4))
        except Exception:
            print("⚠️ Could not generate classification report for these labels.")
        
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        if cm.shape == (2,2):
            print(f"\nTrue Negatives:  {cm[0][0]}")
            print(f"False Positives: {cm[0][1]}")
            print(f"False Negatives: {cm[1][0]}")
            print(f"True Positives:  {cm[1][1]}")
    
    return results

# ========================================
# MODEL COMPARISON
# ========================================
def compare_models(results):
    """Compare all trained models"""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()]
    }).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n", comparison.to_string(index=False))
    
    # Best model
    best_model_name = comparison.iloc[0]['Model']
    best_accuracy = float(comparison.iloc[0]['Accuracy'])
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"✅ Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    return best_model_name, results[best_model_name]['model'], best_accuracy

# ========================================
# MODEL SAVING
# ========================================
def save_models(model, vectorizer, model_name, accuracy=None):
    """Save trained model and vectorizer"""
    print("\n" + "="*60)
    print("💾 SAVING MODELS")
    print("="*60)
    
    model_path = os.path.join(Config.MODEL_PATH, 'fake_news_model.pkl')
    vectorizer_path = os.path.join(Config.MODEL_PATH, 'tfidf_vectorizer.pkl')
    metadata_path = os.path.join(Config.MODEL_PATH, 'metadata.json')
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {model_path}")
    
    # Save vectorizer
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Vectorizer saved: {vectorizer_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'accuracy': accuracy,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")
    
    print(f"\n📝 Model Info:")
    print(f"   - Model Type: {model_name}")
    if accuracy is not None:
        print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - Saved Date: {metadata['date']}")

# ========================================
# TESTING
# ========================================
def test_with_examples(model, vectorizer):
    """Test model with sample news"""
    print("\n" + "="*60)
    print("🧪 TESTING WITH SAMPLE NEWS")
    print("="*60)
    
    test_examples = [
        "Scientists at MIT announce breakthrough in renewable energy technology published in Nature journal",
        "SHOCKING celebrity scandal you wont believe click here now before its deleted",
        "Government releases official statement on new healthcare policy changes nationwide",
        "URGENT share this message now or face consequences limited time offer act fast",
        "Research study shows correlation between diet and mental health outcomes published"
    ]
    
    for i, news in enumerate(test_examples, 1):
        prediction, confidence = predict_news(news, model, vectorizer)
        result = "FAKE NEWS ❌" if prediction == 1 else "REAL NEWS ✅"
        
        print(f"\n{'='*60}")
        print(f"Test {i}:")
        print(f"📰 News: {news[:70]}...")
        print(f"🎯 Prediction: {result}")
        if confidence is not None:
            print(f"📊 Confidence: {confidence:.2f}%")

# ========================================
# MAIN EXECUTION
# ========================================
def main():
    """Main execution function"""
    print("="*60)
    print("🚀 FAKE NEWS DETECTION - TRAINING PIPELINE")
    print("="*60)
    
    try:
        # 1. Setup
        setup_directories()
        setup_nltk()
        
        # 2. Load Data
        df = load_data()
        
        # 3. Explore Data
        df = explore_data(df)
        
        # 4. Preprocess
        df = preprocess_data(df)
        
        # 5. Extract Features
        X, y, vectorizer = extract_features(df)
        
        # 6. Split Data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 7. Train Models
        results = train_all_models(X_train, X_test, y_train, y_test)
        
        # 8. Compare Models
        best_model_name, best_model, best_accuracy = compare_models(results)
        
        # 9. Save Models
        save_models(best_model, vectorizer, best_model_name, accuracy=best_accuracy)
        
        # 10. Test
        test_with_examples(best_model, vectorizer)
        
        # Success
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📌 Next Steps:")
        print("   1. Check models/ folder for saved models")
        print("   2. Run 'streamlit run app.py' for web interface (if available)")
        print("   3. Use the models for predictions")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
# ...existing code..."""
FAKE NEWS DETECTION - MAIN TRAINING SCRIPT
==========================================
Run this file to train the model

Usage:
    python main.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import nltk

# Import custom modules
from preprocessing import preprocess_text, download_nltk_data
from model_training import train_models, evaluate_model
from prediction import predict_news

# ========================================
# CONFIGURATION
# ========================================
class Config:
    """Project configuration"""
    DATA_PATH = 'data/'
    MODEL_PATH = 'models/'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)

# ========================================
# SETUP
# ========================================
def setup_directories():
    """Create necessary directories"""
    os.makedirs(Config.DATA_PATH, exist_ok=True)
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    print("✅ Directories created/verified")

def setup_nltk():
    """Download required NLTK data"""
    print("\n📦 Setting up NLTK data...")
    download_nltk_data()
    print("✅ NLTK data ready")

# ========================================
# DATA LOADING
# ========================================
def load_data():
    """Load dataset"""
    print("\n📂 Loading dataset...")
    
    # Try to load from CSV
    csv_files = [
        'data/fake_news.csv',
        'data/news.csv',
        'fake_news.csv',
        'news.csv'
    ]
    
    for file_path in csv_files:
        if os.path.exists(file_path):
            print(f"✅ Found dataset: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Dataset shape: {df.shape}")
            return df
    
    # If no CSV found, create sample dataset
    print("⚠️ No dataset found. Creating sample dataset...")
    data = create_sample_dataset()
    return data

def create_sample_dataset():
    """Create sample dataset for testing"""
    sample_data = {
        'text': [
            'Scientists discover breakthrough in cancer treatment research published in peer-reviewed journal',
            'SHOCKING: Celebrity caught in scandal you wont believe what happened click here now',
            'Government announces new economic policy to boost infrastructure development',
            'BREAKING: Aliens landed in downtown officials hiding truth from public urgent',
            'Study shows benefits of regular exercise on mental health and wellbeing',
            'Miracle cure discovered doctors hate this one simple trick lose weight fast',
            'Local school receives federal funding for new technology education program',
            'URGENT: Share this message or face bad luck for ten years forward now',
            'Research team publishes comprehensive findings in nature scientific journal',
            'Click here for FREE money limited time offer act now before its gone',
            'University announces new scholarship program for undergraduate students nationwide',
            'You wont believe this secret that celebrities dont want you to know shocking',
            'Federal Reserve maintains interest rates citing economic stability concerns ongoing',
            'AMAZING: This weird trick will change your life forever click to discover',
            'Climate scientists release new report on global warming trends and predictions',
            'Doctors shocked by this one simple method share before its banned forever',
            'Supreme Court issues ruling on landmark case affecting privacy rights nationwide',
            'BREAKING NEWS: Shocking scandal rocks political establishment share immediately urgent',
            'National health institute releases guidelines for preventive care and wellness',
            'Incredible discovery reveals truth about ancient civilization archaeologists stunned amazed',
        ] * 50,  # Repeat to create larger dataset
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 50
    }
    
    df = pd.DataFrame(sample_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"✅ Sample dataset created with {len(df)} articles")
    return df

# ========================================
# DATA EXPLORATION
# ========================================
def explore_data(df):
    """Explore and display dataset information"""
    print("\n" + "="*60)
    print("📊 DATA EXPLORATION")
    print("="*60)
    
    print(f"\n📏 Dataset Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    print("\n🔍 First few rows:")
    print(df.head(3))
    
    print("\n❓ Missing Values:")
    print(df.isnull().sum())
    
    print("\n📊 Class Distribution:")
    if 'label' in df.columns:
        print(df['label'].value_counts())
        print(f"Real News (0): {(df['label']==0).sum()}")
        print(f"Fake News (1): {(df['label']==1).sum()}")
    
    return df

# ========================================
# DATA PREPROCESSING
# ========================================
def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n" + "="*60)
    print("🧹 DATA PREPROCESSING")
    print("="*60)
    
    # Ensure required columns exist
    if 'text' not in df.columns:
        if 'title' in df.columns:
            df['text'] = df['title']
        else:
            raise ValueError("No 'text' or 'title' column found in dataset")
    
    if 'label' not in df.columns:
        raise ValueError("No 'label' column found in dataset")
    
    # Remove missing values
    initial_size = len(df)
    df = df.dropna(subset=['text', 'label'])
    print(f"✅ Removed {initial_size - len(df)} rows with missing values")
    
    # Clean text
    print("🔄 Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Show example
    print("\n📝 Preprocessing Example:")
    print(f"Before: {df['text'].iloc[0][:100]}...")
    print(f"After:  {df['cleaned_text'].iloc[0][:100]}...")
    
    return df

# ========================================
# FEATURE EXTRACTION
# ========================================
def extract_features(df):
    """Extract TF-IDF features"""
    print("\n" + "="*60)
    print("🔢 FEATURE EXTRACTION")
    print("="*60)
    
    print("⚙️ Applying TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(
        max_features=Config.MAX_FEATURES,
        ngram_range=Config.NGRAM_RANGE,
        min_df=2,
        max_df=0.8
    )
    
    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['label']
    
    print(f"✅ TF-IDF Matrix Shape: {X.shape}")
    print(f"✅ Number of Features: {X.shape[1]}")
    print(f"✅ Number of Samples: {X.shape[0]}")
    
    return X, y, tfidf

# ========================================
# TRAIN-TEST SPLIT
# ========================================
def split_data(X, y):
    """Split data into train and test sets"""
    print("\n" + "="*60)
    print("✂️ TRAIN-TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"📚 Training Set: {X_train.shape[0]} samples")
    print(f"📝 Testing Set: {X_test.shape[0]} samples")
    print(f"📊 Split Ratio: {int((1-Config.TEST_SIZE)*100)}% / {int(Config.TEST_SIZE*100)}%")
    
    return X_train, X_test, y_train, y_test

# ========================================
# MODEL TRAINING
# ========================================
def train_all_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare"""
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🔄 Training {name}...")
        print('='*60)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"\n✅ {name} trained successfully!")
        print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Real News', 'Fake News'],
                                    digits=4))
        
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"\nTrue Negatives:  {cm[0][0]}")
        print(f"False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}")
        print(f"True Positives:  {cm[1][1]}")
    
    return results

# ========================================
# MODEL COMPARISON
# ========================================
def compare_models(results):
    """Compare all trained models"""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()]
    }).sort_values('Accuracy', ascending=False)
    
    print("\n", comparison.to_string(index=False))
    
    # Best model
    best_model_name = comparison.iloc[0]['Model']
    best_accuracy = comparison.iloc[0]['Accuracy']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"✅ Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    return best_model_name, results[best_model_name]['model']

# ========================================
# MODEL SAVING
# ========================================
def save_models(model, vectorizer, model_name):
    """Save trained model and vectorizer"""
    print("\n" + "="*60)
    print("💾 SAVING MODELS")
    print("="*60)
    
    model_path = os.path.join(Config.MODEL_PATH, 'fake_news_model.pkl')
    vectorizer_path = os.path.join(Config.MODEL_PATH, 'tfidf_vectorizer.pkl')
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {model_path}")
    
    # Save vectorizer
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Vectorizer saved: {vectorizer_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'accuracy': None,  # Add if needed
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"\n📝 Model Info:")
    print(f"   - Model Type: {model_name}")
    print(f"   - Saved Date: {metadata['date']}")

# ========================================
# TESTING
# ========================================
def test_with_examples(model, vectorizer):
    """Test model with sample news"""
    print("\n" + "="*60)
    print("🧪 TESTING WITH SAMPLE NEWS")
    print("="*60)
    
    test_examples = [
        "Scientists at MIT announce breakthrough in renewable energy technology published in Nature journal",
        "SHOCKING celebrity scandal you wont believe click here now before its deleted",
        "Government releases official statement on new healthcare policy changes nationwide",
        "URGENT share this message now or face consequences limited time offer act fast",
        "Research study shows correlation between diet and mental health outcomes published"
    ]
    
    for i, news in enumerate(test_examples, 1):
        prediction, confidence = predict_news(news, model, vectorizer)
        result = "FAKE NEWS ❌" if prediction == 1 else "REAL NEWS ✅"
        
        print(f"\n{'='*60}")
        print(f"Test {i}:")
        print(f"📰 News: {news[:70]}...")
        print(f"🎯 Prediction: {result}")
        if confidence:
            print(f"📊 Confidence: {confidence:.2f}%")

# ========================================
# MAIN EXECUTION
# ========================================
def main():
    """Main execution function"""
    print("="*60)
    print("🚀 FAKE NEWS DETECTION - TRAINING PIPELINE")
    print("="*60)
    
    try:
        # 1. Setup
        setup_directories()
        setup_nltk()
        
        # 2. Load Data
        df = load_data()
        
        # 3. Explore Data
        df = explore_data(df)
        
        # 4. Preprocess
        df = preprocess_data(df)
        
        # 5. Extract Features
        X, y, vectorizer = extract_features(df)
        
        # 6. Split Data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 7. Train Models
        results = train_all_models(X_train, X_test, y_train, y_test)
        
        # 8. Compare Models
        best_model_name, best_model = compare_models(results)
        
        # 9. Save Models
        save_models(best_model, vectorizer, best_model_name)
        
        # 10. Test
        test_with_examples(best_model, vectorizer)
        
        # Success
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📌 Next Steps:")
        print("   1. Check models/ folder for saved models")
        print("   2. Run 'streamlit run app.py' for web interface")
        print("   3. Use the models for predictions")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())