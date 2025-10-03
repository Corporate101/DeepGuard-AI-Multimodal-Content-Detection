import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class TextDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
        """Load or initialize the text detection model"""
        model_path = "models/text_detector_model.pkl"
        vectorizer_path = "models/text_vectorizer.pkl"
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            # Initialize with a simple model for demonstration
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self._train_demo_model()
    
    def _train_demo_model(self):
        """Train a simple demo model with synthetic data"""
        # This would be replaced with real training data
        human_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "I went to the store to buy some groceries for dinner.",
            "The weather today is quite pleasant with a gentle breeze.",
            "She completed her assignment and submitted it before the deadline.",
            "We decided to take a walk in the park after lunch."
        ]
        
        ai_texts = [
            "The aforementioned canine entity was observed traversing the terrain.",
            "Subsequent to evaluating available options, procurement decisions were made.",
            "Meteorological conditions presently exhibit favorable characteristics.",
            "The assigned task was finalized and transmitted prior to the stipulated timeframe.",
            "A decision was reached to engage in pedestrian locomotion within the recreational area."
        ]
        
        texts = human_texts + ai_texts
        labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0=human, 1=AI
        
        # Transform and train
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        
        # Save models
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/text_detector_model.pkl")
        joblib.dump(self.vectorizer, "models/text_vectorizer.pkl")
    
    def analyze_text(self, text):
        """Analyze text for AI generation indicators"""
        start_time = time.time()
        
        if not text or len(text.strip()) < 50:
            return {
                'is_ai_generated': False,
                'confidence': 0.5,
                'error': 'Text too short for reliable analysis',
                'processing_time': time.time() - start_time
            }
        
        try:
            # Extract features
            features = self._extract_text_features(text)
            
            # Transform text for model prediction
            X = self.vectorizer.transform([text])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[0]
            ai_probability = probabilities[1]  # Probability of being AI-generated
            
            # Determine result
            confidence = abs(ai_probability - 0.5) * 2  # Convert to confidence score
            is_ai = ai_probability > 0.6  # Threshold for AI detection
            
            return {
                'is_ai_generated': is_ai,
                'confidence': confidence,
                'ai_probability': ai_probability,
                'features': features,
                'processing_time': time.time() - start_time,
                'text_length': len(text)
            }
            
        except Exception as e:
            return {
                'is_ai_generated': False,
                'confidence': 0.5,
                'error': f'Analysis error: {str(e)}',
                'processing_time': time.time() - start_time
            }
    
    def _extract_text_features(self, text):
        """Extract various text features for analysis"""
        # Basic text statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'punctuation_density': len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        }
        
        # Readability scores (simplified)
        if sentences and words:
            features['readability_score'] = min(100, max(0, 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (sum(len(word) for word in words) / len(words))))
        else:
            features['readability_score'] = 50
        
        return features
