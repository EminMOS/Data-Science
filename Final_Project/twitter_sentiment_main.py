#!/usr/bin/env python3
"""
Twitter Sentiment Analysis Using Classical & Neural Models
============================================================
This script implements sentiment classification on Twitter data using the Kaggle 
"Twitter Sentiment Dataset" comparing classical ML models against a CNN-LSTM hybrid.

Author: Twitter Sentiment Analysis Pipeline
Date: 2024
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os

# Text processing
import re
import string

# Feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# Classical ML models
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# Evaluation metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

# Neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Conv1D, MaxPooling1D, 
                                    LSTM, Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Resolve project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, 'Twitter_Data.csv')
DEFAULT_SAMPLE_SIZE = 100

# Set style and warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Basic English stopwords list (instead of NLTK)
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def get_num_samples(data):
    """Return the number of samples for dense or sparse matrices."""
    shape = getattr(data, 'shape', None)
    if shape is not None:
        try:
            return shape[0]
        except (TypeError, IndexError):
            pass
    try:
        return len(data)
    except TypeError as exc:
        raise TypeError(f"Unable to determine sample size for object of type {type(data)}") from exc

class TwitterSentimentAnalyzer:
    """Main class for Twitter Sentiment Analysis Pipeline"""
    
    def __init__(self, data_path, random_state=42, sample_size=None, development_mode=False):
        """
        Initialize the analyzer with data path and random state
        
        Args:
            data_path: Path to the Twitter dataset CSV file
            random_state: Random state for reproducibility
            sample_size: Number of samples to use (None for full dataset)
            development_mode: If True, automatically uses small sample for fast testing
        """
        self.data_path = data_path
        self.random_state = random_state
        self.full_dataset_requested = bool(sample_size is not None and sample_size <= 0)
        self.sample_size = None if self.full_dataset_requested else sample_size
        self.development_mode = development_mode
        self.df = None
        self.df_full = None  # Store full dataset
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.stop_words = ENGLISH_STOPWORDS
        
        # If development mode, set default sample size
        if self.development_mode and self.sample_size is None and not self.full_dataset_requested:
            self.sample_size = 1000  # Default for dev mode
        
    def load_data(self):
        """Load the Twitter dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(self.df.head(3))
        
        return self.df
    
    def sample_dataset(self, force_sample_size=None):
        """
        Sample the dataset for faster processing during development/testing
        
        Args:
            force_sample_size: Override the instance sample_size if provided
            
        Returns:
            None (modifies self.df in place)
        """
        sample_size = force_sample_size or self.sample_size
        
        if sample_size is None:
            print("\nUsing FULL dataset for training")
            return
        
        # Store the full dataset if not already stored
        if self.df_full is None:
            self.df_full = self.df.copy()
        
        # Ensure we don't sample more than available
        actual_sample_size = min(sample_size, len(self.df))
        
        print("\n" + "=" * 80)
        print("DATASET SAMPLING FOR FASTER PROCESSING")
        print("=" * 80)
        print(f"Original dataset size: {len(self.df):,} samples")
        print(f"Sampling to: {actual_sample_size:,} samples")
        
        # Drop missing values first to ensure stratified sampling works
        df_clean = self.df.dropna(subset=['category'])
        df_clean['category'] = df_clean['category'].astype(int)
        
        # Check if we have enough samples per class for stratified sampling
        min_class_count = df_clean['category'].value_counts().min()
        
        if actual_sample_size > len(df_clean):
            actual_sample_size = len(df_clean)
            print(f"Adjusted sample size to available data: {actual_sample_size:,}")
        
        # Use stratified sampling to maintain class distribution
        if min_class_count >= 3 and actual_sample_size >= 30:
            # Can do stratified sampling
            from sklearn.model_selection import train_test_split
            _, self.df = train_test_split(
                df_clean, 
                test_size=actual_sample_size/len(df_clean),
                stratify=df_clean['category'],
                random_state=self.random_state
            )
            print("✓ Used stratified sampling to maintain class distribution")
        else:
            # Simple random sampling
            self.df = df_clean.sample(n=actual_sample_size, random_state=self.random_state)
            print("✓ Used random sampling (dataset too small for stratified sampling)")
        
        # Show new distribution
        print("\nSampled dataset distribution:")
        sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        for cat in [-1, 0, 1]:
            count = (self.df['category'] == cat).sum()
            if count > 0:
                percentage = (count / len(self.df)) * 100
                print(f"   {sentiment_map[cat]:10s}: {count:6d} ({percentage:5.2f}%)")
        
        print(f"\nNew working dataset shape: {self.df.shape}")
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Basic info
        print("\n1. Dataset Info:")
        print(self.df.info())
        
        # Missing values
        print(f"\n2. Missing values:\n{self.df.isnull().sum()}")
        
        # Class distribution
        print("\n3. Sentiment Distribution:")
        sentiment_counts = self.df['category'].value_counts().sort_index()
        sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        
        for cat, count in sentiment_counts.items():
            cat_int = int(cat)  # Convert float to int for display
            percentage = (count / len(self.df)) * 100
            print(f"   {sentiment_map.get(cat_int, 'Unknown'):10s} ({cat_int:2d}): {count:6d} ({percentage:5.2f}%)")
        
        # Text length analysis
        self.df['text_length'] = self.df['clean_text'].fillna('').apply(len)
        self.df['word_count'] = self.df['clean_text'].fillna('').apply(lambda x: len(str(x).split()))
        
        print("\n4. Text Statistics:")
        print(f"   Average text length: {self.df['text_length'].mean():.2f} characters")
        print(f"   Average word count: {self.df['word_count'].mean():.2f} words")
        print(f"   Max text length: {self.df['text_length'].max()} characters")
        print(f"   Min text length: {self.df['text_length'].min()} characters")
        
        # Visualizations
        self.create_eda_visualizations()
        
    def create_eda_visualizations(self):
        """Create EDA visualizations"""
        # First, drop rows with missing categories for visualization
        df_viz = self.df.dropna(subset=['category']).copy()
        df_viz['category'] = df_viz['category'].astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sentiment distribution
        sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        df_viz['sentiment_label'] = df_viz['category'].map(sentiment_map)
        
        ax1 = axes[0, 0]
        df_viz['sentiment_label'].value_counts().plot(kind='bar', ax=ax1, color=['#e74c3c', '#95a5a6', '#2ecc71'])
        ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Text length distribution by sentiment
        ax2 = axes[0, 1]
        for sentiment in ['Negative', 'Neutral', 'Positive']:
            subset = df_viz[df_viz['sentiment_label'] == sentiment]['text_length']
            ax2.hist(subset, alpha=0.5, label=sentiment, bins=50)
        ax2.set_title('Text Length Distribution by Sentiment', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Text Length')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Word count distribution
        ax3 = axes[1, 0]
        ax3.hist(df_viz['word_count'], bins=50, color='skyblue', edgecolor='black')
        ax3.set_title('Word Count Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Word Count')
        ax3.set_ylabel('Frequency')
        ax3.axvline(df_viz['word_count'].mean(), color='red', linestyle='--', label=f'Mean: {df_viz["word_count"].mean():.0f}')
        ax3.legend()
        
        # Box plot of text length by sentiment
        ax4 = axes[1, 1]
        df_viz.boxplot(column='text_length', by='sentiment_label', ax=ax4)
        ax4.set_title('Text Length by Sentiment (Box Plot)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Sentiment')
        ax4.set_ylabel('Text Length')
        plt.sca(ax4)
        plt.xticks(rotation=45)
        
        plt.suptitle('')  # Remove the default title
        plt.tight_layout()
        plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_text(self, text):
        """
        Preprocess individual text
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize (simple split-based tokenization)
        tokens = text.split()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_data(self):
        """Preprocess the entire dataset"""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        
        # Handle missing values
        print("Handling missing values...")
        self.df['clean_text'] = self.df['clean_text'].fillna('')
        # Drop rows with missing categories
        self.df = self.df.dropna(subset=['category'])
        # Convert category to int
        self.df['category'] = self.df['category'].astype(int)
        
        # Apply preprocessing
        print("Preprocessing text (this may take a while)...")
        self.df['processed_text'] = self.df['clean_text'].apply(self.preprocess_text)
        
        # Remove empty texts after preprocessing
        original_len = len(self.df)
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        removed = original_len - len(self.df)
        print(f"Removed {removed} empty texts after preprocessing")
        
        print(f"Final dataset shape: {self.df.shape}")
        
    def prepare_tfidf_features(self):
        """Prepare TF-IDF features"""
        print("\n" + "=" * 80)
        print("PREPARING TF-IDF FEATURES")
        print("=" * 80)
        
        # Adjust max_features based on dataset size
        dataset_size = len(self.df)
        if self.development_mode or dataset_size <= 1000:
            max_features = 500
            print(f"📝 Using {max_features} features (small dataset mode)")
        elif dataset_size <= 5000:
            max_features = 1000
            print(f"📝 Using {max_features} features (medium dataset mode)")
        else:
            max_features = 5000
            print(f"📝 Using {max_features} features (full dataset mode)")
        
        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2 if dataset_size <= 1000 else 5,
            max_df=0.95
        )
        
        # Split data first
        X = self.df['processed_text']
        y = self.df['category']
        
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Transform to TF-IDF
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train_text)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test_text)
        
        print(f"TF-IDF training features shape: {X_train_tfidf.shape}")
        print(f"TF-IDF test features shape: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
    def prepare_word2vec_features(self):
        """Prepare Word2Vec features"""
        print("\n" + "=" * 80)
        print("PREPARING WORD2VEC FEATURES")
        print("=" * 80)
        
        # Tokenize texts for Word2Vec
        tokenized_texts = [text.split() for text in self.df['processed_text']]
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=3,
            workers=4,
            seed=self.random_state
        )
        
        # Create average word vectors for each tweet
        def get_avg_word2vec(tokens, model, vector_size):
            """Get average Word2Vec representation for a list of tokens"""
            vec = np.zeros(vector_size)
            count = 0
            for word in tokens:
                if word in model.wv:
                    vec += model.wv[word]
                    count += 1
            if count > 0:
                vec /= count
            return vec
        
        print("Creating document vectors...")
        X_word2vec = np.array([
            get_avg_word2vec(text.split(), self.word2vec_model, 100) 
            for text in self.df['processed_text']
        ])
        
        # Split data
        y = self.df['category'].values
        X_train_w2v, X_test_w2v, y_train, y_test = train_test_split(
            X_word2vec, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Word2Vec training features shape: {X_train_w2v.shape}")
        print(f"Word2Vec test features shape: {X_test_w2v.shape}")
        
        return X_train_w2v, X_test_w2v, y_train, y_test
    
    def train_classical_models(self, X_train, X_test, y_train, y_test, feature_type):
        """
        Train classical ML models with hyperparameter tuning
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Labels
            feature_type: Type of features ('tfidf' or 'word2vec')
        """
        print(f"\n" + "=" * 80)
        print(f"TRAINING CLASSICAL MODELS WITH {feature_type.upper()} FEATURES")
        print("=" * 80)
        
        results = {}
        
        # Determine if we should use simplified parameters for small datasets
        use_simple_params = (
            self.development_mode or 
            (self.sample_size is not None and self.sample_size <= 5000) or
            get_num_samples(X_train) <= 5000
        )
        
        if use_simple_params:
            print("📝 Using simplified hyperparameters for faster execution")
            
        # 1. Decision Tree
        print("\n1. Training Decision Tree with GridSearchCV...")
        
        if use_simple_params:
            # Simplified parameters for small datasets
            dt_params = {
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'criterion': ['gini', 'entropy']
            }
        else:
            # Full parameters for large datasets
            dt_params = {
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        
        dt_grid = GridSearchCV(
            DecisionTreeClassifier(random_state=self.random_state),
            dt_params,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        dt_grid.fit(X_train, y_train)
        
        dt_best = dt_grid.best_estimator_
        dt_pred = dt_best.predict(X_test)
        
        results['DecisionTree'] = {
            'model': dt_best,
            'predictions': dt_pred,
            'best_params': dt_grid.best_params_,
            'accuracy': accuracy_score(y_test, dt_pred),
            'f1_macro': f1_score(y_test, dt_pred, average='macro')
        }
        print(f"   Best params: {dt_grid.best_params_}")
        print(f"   Accuracy: {results['DecisionTree']['accuracy']:.4f}")
        print(f"   F1-macro: {results['DecisionTree']['f1_macro']:.4f}")
        
        # 2. K-Nearest Neighbors
        print("\n2. Training KNN with GridSearchCV...")
        
        if use_simple_params:
            # Simplified parameters for small datasets
            knn_params = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        else:
            # Full parameters for large datasets
            knn_params = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        
        knn_grid = GridSearchCV(
            KNeighborsClassifier(),
            knn_params,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        knn_grid.fit(X_train, y_train)
        
        knn_best = knn_grid.best_estimator_
        knn_pred = knn_best.predict(X_test)
        
        results['KNN'] = {
            'model': knn_best,
            'predictions': knn_pred,
            'best_params': knn_grid.best_params_,
            'accuracy': accuracy_score(y_test, knn_pred),
            'f1_macro': f1_score(y_test, knn_pred, average='macro')
        }
        print(f"   Best params: {knn_grid.best_params_}")
        print(f"   Accuracy: {results['KNN']['accuracy']:.4f}")
        print(f"   F1-macro: {results['KNN']['f1_macro']:.4f}")
        
        # 3. Logistic Regression
        print("\n3. Training Logistic Regression with GridSearchCV...")
        
        if use_simple_params:
            # Simplified parameters for small datasets
            lr_params = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [100]
            }
        else:
            # Full parameters for large datasets
            lr_params = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [100, 200, 500]
            }
        
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=self.random_state, multi_class='ovr'),
            lr_params,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        lr_grid.fit(X_train, y_train)
        
        lr_best = lr_grid.best_estimator_
        lr_pred = lr_best.predict(X_test)
        
        results['LogisticRegression'] = {
            'model': lr_best,
            'predictions': lr_pred,
            'best_params': lr_grid.best_params_,
            'accuracy': accuracy_score(y_test, lr_pred),
            'f1_macro': f1_score(y_test, lr_pred, average='macro')
        }
        print(f"   Best params: {lr_grid.best_params_}")
        print(f"   Accuracy: {results['LogisticRegression']['accuracy']:.4f}")
        print(f"   F1-macro: {results['LogisticRegression']['f1_macro']:.4f}")
        
        # 4. Voting Classifier (Ensemble)
        print("\n4. Training Voting Classifier (Ensemble)...")
        voting_clf = VotingClassifier(
            estimators=[
                ('dt', dt_best),
                ('knn', knn_best),
                ('lr', lr_best)
            ],
            voting='soft' if hasattr(knn_best, "predict_proba") else 'hard'
        )
        
        voting_clf.fit(X_train, y_train)
        voting_pred = voting_clf.predict(X_test)
        
        results['VotingClassifier'] = {
            'model': voting_clf,
            'predictions': voting_pred,
            'accuracy': accuracy_score(y_test, voting_pred),
            'f1_macro': f1_score(y_test, voting_pred, average='macro')
        }
        print(f"   Accuracy: {results['VotingClassifier']['accuracy']:.4f}")
        print(f"   F1-macro: {results['VotingClassifier']['f1_macro']:.4f}")
        
        return results
    
    def prepare_neural_network_data(self):
        """Prepare data for neural network"""
        print("\n" + "=" * 80)
        print("PREPARING DATA FOR NEURAL NETWORK")
        print("=" * 80)
        
        # Adjust parameters based on dataset size
        dataset_size = len(self.df)
        if self.development_mode or dataset_size <= 1000:
            max_words = 1000
            max_len = 50
            print(f"📝 Using simplified NN parameters (vocab={max_words}, max_len={max_len})")
        elif dataset_size <= 5000:
            max_words = 5000
            max_len = 75
            print(f"📝 Using medium NN parameters (vocab={max_words}, max_len={max_len})")
        else:
            max_words = 10000
            max_len = 100
            print(f"📝 Using full NN parameters (vocab={max_words}, max_len={max_len})")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=max_words)
        self.tokenizer.fit_on_texts(self.df['processed_text'])
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(self.df['processed_text'])
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=max_len)
        
        # Convert labels to categorical (shift -1,0,1 to 0,1,2)
        y = self.df['category'].values + 1  # Shift to 0,1,2
        y_categorical = to_categorical(y, num_classes=3)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Neural network training data shape: {X_train.shape}")
        print(f"Neural network test data shape: {X_test.shape}")
        print(f"Number of unique tokens: {len(self.tokenizer.word_index)}")
        
        return X_train, X_test, y_train, y_test, max_words, max_len
    
    def build_cnn_lstm_model(self, max_words, max_len):
        """
        Build CNN-LSTM hybrid model
        
        Args:
            max_words: Maximum number of words in vocabulary
            max_len: Maximum sequence length
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Embedding layer
            Embedding(max_words, 128, input_length=max_len),
            
            # CNN layers
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # LSTM layer
            LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
            
            # Dense layers
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            # Output layer
            Dense(3, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_neural_network(self, X_train, X_test, y_train, y_test, max_words, max_len):
        """Train CNN-LSTM neural network"""
        print("\n" + "=" * 80)
        print("TRAINING CNN-LSTM NEURAL NETWORK")
        print("=" * 80)
        
        # Adjust parameters based on dataset size
        dataset_size = get_num_samples(X_train)
        if self.development_mode or dataset_size <= 1000:
            epochs = 3
            batch_size = 16
            print("📝 Using simplified neural network training (3 epochs, batch_size=16)")
        elif dataset_size <= 5000:
            epochs = 5
            batch_size = 32
            print("📝 Using medium neural network training (5 epochs, batch_size=32)")
        else:
            epochs = 10
            batch_size = 32
            print("📝 Using full neural network training (10 epochs, batch_size=32)")
        
        # Build model
        model = self.build_cnn_lstm_model(max_words, max_len)
        
        print("\nModel Architecture:")
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2 if dataset_size <= 5000 else 3, 
                         restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
        ]
        
        # Train model
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1-Macro: {f1_macro:.4f}")
        
        # Plot training history
        self.plot_training_history(history)
        
        results = {
            'model': model,
            'history': history,
            'predictions': y_pred - 1,  # Shift back to -1,0,1
            'true_labels': y_true - 1,  # Shift back to -1,0,1
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'loss': loss
        }
        
        return results
    
    def plot_training_history(self, history):
        """Plot neural network training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Comprehensive evaluation of a model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS FOR {model_name}")
        print('='*60)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Macro:  {f1_macro:.4f}")
        
        # Per-class metrics
        print(f"\nClassification Report:")
        sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        target_names = [sentiment_map[i] for i in sorted(np.unique(y_true))]
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'confusion_matrix': cm
        }
    
    def plot_roc_curves(self, models_results, y_test):
        """
        Plot ROC curves for all models
        
        Args:
            models_results: Dictionary of model results
            y_test: True test labels
        """
        print("\n" + "=" * 80)
        print("PLOTTING ROC CURVES")
        print("=" * 80)
        
        plt.figure(figsize=(12, 8))
        
        # Binarize labels for multi-class ROC
        y_test_bin = label_binarize(y_test + 1, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            if 'model' not in results:
                continue
                
            model = results['model']
            
            # Get probability predictions if available
            if hasattr(model, "predict_proba"):
                # Need to transform test data appropriately
                if 'tfidf' in model_name.lower():
                    X_test_transformed = self.tfidf_vectorizer.transform(
                        self.df['processed_text'].iloc[-len(y_test):]
                    )
                else:
                    continue  # Skip if we can't get proper test data
                    
                y_score = model.predict_proba(X_test_transformed)
                
                # Compute ROC curve and AUC for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Compute micro-average ROC curve
                fpr["micro"], tpr["micro"], _ = roc_curve(
                    y_test_bin.ravel(), y_score.ravel()
                )
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                # Plot micro-average ROC
                plt.plot(fpr["micro"], tpr["micro"],
                        label=f'{model_name} (AUC = {roc_auc["micro"]:.3f})',
                        color=colors[idx % len(colors)], linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_table(self, all_results):
        """
        Create a comparison table of all models
        
        Args:
            all_results: Dictionary containing all model results
        """
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        
        comparison_data = []
        
        for feature_type, models in all_results.items():
            for model_name, metrics in models.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    comparison_data.append({
                        'Feature Type': feature_type,
                        'Model': model_name,
                        'Accuracy': metrics.get('accuracy', 0),
                        'F1-Macro': metrics.get('f1_macro', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0)
                    })
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Macro', ascending=False)
        
        # Format for display
        for col in ['Accuracy', 'F1-Macro', 'Precision', 'Recall']:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.4f}")
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Save to CSV
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("\nComparison table saved to 'model_comparison.csv'")
        
        # Create visualization
        self.plot_model_comparison(comparison_data)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_data):
        """Create bar plot comparing model performances"""
        df = pd.DataFrame(comparison_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy comparison
        ax1.barh(range(len(df)), df['Accuracy'], color='skyblue')
        ax1.set_yticks(range(len(df)))
        ax1.set_yticklabels([f"{row['Model']} ({row['Feature Type']})" 
                             for _, row in df.iterrows()])
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(df['Accuracy']):
            ax1.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        # F1-Macro comparison
        ax2.barh(range(len(df)), df['F1-Macro'], color='lightcoral')
        ax2.set_yticks(range(len(df)))
        ax2.set_yticklabels([f"{row['Model']} ({row['Feature Type']})" 
                             for _, row in df.iterrows()])
        ax2.set_xlabel('F1-Macro Score')
        ax2.set_title('Model Comparison - F1-Macro', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(df['F1-Macro']):
            ax2.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, all_results):
        """Save trained models and vectorizers"""
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        print("Saved TF-IDF vectorizer")
        
        # Save Word2Vec model
        if hasattr(self, 'word2vec_model'):
            self.word2vec_model.save('models/word2vec_model.bin')
            print("Saved Word2Vec model")
        
        # Save tokenizer for neural network
        if hasattr(self, 'tokenizer'):
            with open('models/nn_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print("Saved neural network tokenizer")
        
        # Save classical models
        for feature_type, models in all_results.items():
            if feature_type == 'neural_network':
                continue
            for model_name, results in models.items():
                if 'model' in results:
                    filename = f'models/{model_name}_{feature_type}.pkl'
                    with open(filename, 'wb') as f:
                        pickle.dump(results['model'], f)
                    print(f"Saved {model_name} with {feature_type} features")
        
        print("\nAll models saved successfully!")
    
    def run_pipeline(self):
        """Run the complete sentiment analysis pipeline"""
        print("\n" + "=" * 80)
        print("TWITTER SENTIMENT ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Load data
        self.load_data()
        
        # 1.5. Sample dataset if specified (for faster development/testing)
        self.sample_dataset()
        
        # 2. Perform EDA
        self.perform_eda()
        
        # 3. Preprocess data
        self.preprocess_data()
        
        # Store all results
        all_results = {}
        
        # 4. TF-IDF Features Pipeline
        print("\n" + "#" * 80)
        print("TFIDF FEATURE PIPELINE")
        print("#" * 80)
        X_train_tfidf, X_test_tfidf, y_train, y_test = self.prepare_tfidf_features()
        tfidf_results = self.train_classical_models(
            X_train_tfidf, X_test_tfidf, y_train, y_test, 'tfidf'
        )
        
        # Evaluate each TF-IDF model
        for model_name, results in tfidf_results.items():
            eval_metrics = self.evaluate_model(
                y_test, results['predictions'], f"{model_name} (TF-IDF)"
            )
            tfidf_results[model_name].update(eval_metrics)
        
        all_results['TF-IDF'] = tfidf_results
        
        # 5. Word2Vec Features Pipeline
        print("\n" + "#" * 80)
        print("WORD2VEC FEATURE PIPELINE")
        print("#" * 80)
        X_train_w2v, X_test_w2v, y_train, y_test = self.prepare_word2vec_features()
        w2v_results = self.train_classical_models(
            X_train_w2v, X_test_w2v, y_train, y_test, 'word2vec'
        )
        
        # Evaluate each Word2Vec model
        for model_name, results in w2v_results.items():
            eval_metrics = self.evaluate_model(
                y_test, results['predictions'], f"{model_name} (Word2Vec)"
            )
            w2v_results[model_name].update(eval_metrics)
        
        all_results['Word2Vec'] = w2v_results
        
        # 6. Neural Network Pipeline
        print("\n" + "#" * 80)
        print("NEURAL NETWORK PIPELINE")
        print("#" * 80)
        X_train_nn, X_test_nn, y_train_nn, y_test_nn, max_words, max_len = self.prepare_neural_network_data()
        nn_results = self.train_neural_network(
            X_train_nn, X_test_nn, y_train_nn, y_test_nn, max_words, max_len
        )
        
        # Evaluate neural network
        eval_metrics = self.evaluate_model(
            nn_results['true_labels'], nn_results['predictions'], "CNN-LSTM Neural Network"
        )
        nn_results.update(eval_metrics)
        
        all_results['Neural Network'] = {'CNN-LSTM': nn_results}
        
        # 7. Create comparison table
        comparison_df = self.create_comparison_table(all_results)
        
        # 8. Plot ROC curves (for models with probability predictions)
        self.plot_roc_curves(tfidf_results, y_test)
        
        # 9. Save models
        self.save_models(all_results)
        
        # 10. Final summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nGenerated files:")
        print("  - eda_visualization.png")
        print("  - nn_training_history.png")
        print("  - confusion_matrix_*.png (for each model)")
        print("  - roc_curves.png")
        print("  - model_comparison.png")
        print("  - model_comparison.csv")
        print("  - models/ directory with saved models")
        print("  - best_model.h5 (best neural network weights)")
        
        # Identify best model
        print("\n" + "=" * 80)
        print("BEST MODEL")
        print("=" * 80)
        best_model_idx = comparison_df.iloc[0]
        print(f"Best performing model: {best_model_idx['Model']} with {best_model_idx['Feature Type']} features")
        print(f"  - Accuracy: {best_model_idx['Accuracy']}")
        print(f"  - F1-Macro: {best_model_idx['F1-Macro']}")
        
        return all_results, comparison_df

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Twitter Sentiment Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with full dataset (default)
  python twitter_sentiment_main.py
  
  # Run with 100 samples for quick testing
  python twitter_sentiment_main.py --sample-size 100
  
  # Run in development mode (1000 samples, simplified parameters)
  python twitter_sentiment_main.py --dev
  
  # Run with custom sample size
  python twitter_sentiment_main.py --sample-size 5000
        """
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help='Path to the Twitter dataset CSV file'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of samples to use (default: 100, set <=0 to use full dataset)'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Development mode: uses 1000 samples and simplified parameters for fast testing'
    )
    
    args = parser.parse_args()
    effective_sample_size = args.sample_size
    full_dataset_requested = effective_sample_size is not None and effective_sample_size <= 0
    if full_dataset_requested:
        effective_sample_size = None
    
    # Print configuration
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    
    if args.dev:
        print("DEVELOPMENT MODE ENABLED")
        print("   - Default sample size: 1,000 (unless overridden)")
        print("   - Simplified hyperparameters for faster execution")
        if not full_dataset_requested and effective_sample_size is None:
            effective_sample_size = 1000
        if full_dataset_requested:
            print("FULL DATASET MODE REQUESTED")
        else:
            print(f"SAMPLE MODE: Using {effective_sample_size:,} samples")
    else:
        if full_dataset_requested:
            print("FULL DATASET MODE")
        else:
            if effective_sample_size is None:
                effective_sample_size = DEFAULT_SAMPLE_SIZE
                print(f"DEFAULT SAMPLE MODE: Using {effective_sample_size:,} samples")
            else:
                print(f"SAMPLE MODE: Using {effective_sample_size:,} samples")
    
    # Initialize analyzer
    analyzer = TwitterSentimentAnalyzer(
        data_path=args.data_path,
        random_state=args.random_state,
        sample_size=effective_sample_size,
        development_mode=args.dev
    )
    
    # Run pipeline
    try:
        all_results, comparison_df = analyzer.run_pipeline()
        print("\n✅ Pipeline executed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
