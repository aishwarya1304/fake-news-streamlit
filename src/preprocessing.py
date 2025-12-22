"""
PREPROCESSING MODULE
===================
Text preprocessing functions for fake news detection
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text for ML model
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join back
    return ' '.join(tokens)

def contains_spam_words(text):
    """
    Check if text contains spam/clickbait words
    
    Args:
        text (str): Input text
        
    Returns:
        bool: True if contains spam words
    """
    spam_words = [
        'shocking', 'breaking', 'urgent', 'click here', 'limited time',
        'act now', 'you won\'t believe', 'amazing', 'incredible',
        'miracle', 'secret', 'doctors hate', 'banned forever',
        'share this', 'forward now', 'bad luck', 'free money',
        'one weird trick', 'celebrities reveal', 'doctors shocked'
    ]
    
    text_lower = text.lower()
    return any(spam_word in text_lower for spam_word in spam_words)