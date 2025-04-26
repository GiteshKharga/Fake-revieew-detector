import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from utils import load_yelp_data, calculate_metrics
from textblob import TextBlob
from tqdm import tqdm
import time
import os
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import json

# Ensure NLTK data directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK data with explicit paths
print("Downloading NLTK data...")
for package in ['stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.download(package, quiet=True, raise_on_error=True)
    except Exception as e:
        print(f"Error downloading {package}: {str(e)}")
        print("Retrying download...")
        nltk.download(package, quiet=False)

# Verify NLTK data
print("Verifying NLTK data...")
try:
    # Test stopwords
    stopwords.words('english')
    print("NLTK data verification successful!")
except Exception as e:
    print(f"NLTK data verification failed: {str(e)}")
    raise

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_features(self, text):
        """Extract additional features from text"""
        # Basic text statistics
        words = text.split()
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if word_count > 0 else 0
        unique_words = len(set(words))
        chars_count = len(text)
        
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        subjectivity_score = blob.sentiment.subjectivity
        
        # Text complexity features
        words_per_char = word_count / chars_count if chars_count > 0 else 0
        unique_word_ratio = unique_words / word_count if word_count > 0 else 0
        
        # Capitalization features
        capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # Punctuation features
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        punctuation_ratio = punctuation_count / len(text) if len(text) > 0 else 0
        
        return {
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'unique_words': unique_words,
            'chars_count': chars_count,
            'sentiment_score': sentiment_score,
            'subjectivity_score': subjectivity_score,
            'words_per_char': words_per_char,
            'unique_word_ratio': unique_word_ratio,
            'capital_ratio': capital_ratio,
            'punctuation_ratio': punctuation_ratio
        }
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        
        # Lemmatization
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def load_data_in_chunks(self, chunk_size=5000):
        """Load and process data in chunks to manage memory"""
        print("\nLoading dataset in chunks...")
        texts = []
        labels = []
        processed_texts = []
        features = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading data")):
                if i >= 25000:  # Limit to 25000 reviews
                    break
                    
                try:
                    review = json.loads(line)
                    texts.append(review['text'])
                    labels.append(1 if review['stars'] >= 4 else 0)
                except json.JSONDecodeError:
                    continue
                
                # Process in chunks
                if len(texts) >= chunk_size:
                    # Preprocess texts
                    chunk_processed = [self.preprocess_text(text) for text in texts]
                    processed_texts.extend(chunk_processed)
                    
                    # Extract features
                    chunk_features = [self.extract_text_features(text) for text in texts]
                    features.extend(chunk_features)
                    
                    # Clear memory
                    texts = []
            
            # Process remaining texts
            if texts:
                chunk_processed = [self.preprocess_text(text) for text in texts]
                processed_texts.extend(chunk_processed)
                chunk_features = [self.extract_text_features(text) for text in texts]
                features.extend(chunk_features)
        
        return processed_texts, labels, features
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for training with enhanced features"""
        # Load and process data in chunks
        processed_texts, labels, features = self.load_data_in_chunks()
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame(features)
        
        print("\nCreating TF-IDF features...")
        tfidf = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        # Create TF-IDF features
        tfidf_features = tfidf.fit_transform(processed_texts)
        
        # Scale additional features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        
        # Combine TF-IDF and additional features
        X = hstack([tfidf_features, scaled_features])
        y = np.array(labels)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, tfidf, scaler

if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor("dataset/yelp_academic_dataset_review.json")
    X_train, X_test, y_train, y_test, tfidf, scaler = processor.prepare_data() 