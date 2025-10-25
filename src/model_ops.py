# src/model_ops.py
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- CRITICAL FIX APPLIED: Used RELATIVE import (the dot '.') for sibling files ---
from .db_ops import fetch_training_data, update_training_status
from .preprocess import clean_text

MODEL_PATH = 'app/models/spam_model.pkl' 

def train_model():
    """Loads data from DB, trains the pipeline, and saves the model."""
    try:
        df = fetch_training_data()
        
        # Handle cases where not enough data is available
        if df.shape[0] == 0: 
            return "Training skipped: No data available. Please add samples via the UI."
            
        if df.shape[0] < 5: 
            return f"Training skipped: Only {df.shape[0]} samples available (need at least 5 for split)."

        if len(df['true_label'].unique()) < 2:
            return "Training skipped: Need samples for both 'ham' and 'spam' classes."
            
        # 1. Preprocessing and preparation
        df['cleaned_text'] = df['message_text'].apply(clean_text)
        
        X = df['cleaned_text']
        y = df['true_label'].map({'ham': 0, 'spam': 1})
        sample_weights = df['priority'].values
        
        # 2. Define ML Pipeline
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('classifier', MultinomialNB()),
        ])
        
        # 3. Training and Evaluation Split
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        model_pipeline.fit(X_train, y_train, classifier__sample_weight=w_train)
        
        # 4. Evaluation and Final Saving
        y_pred = model_pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        
        # Final Training on ALL data
        model_pipeline.fit(X, y, classifier__sample_weight=sample_weights)
        joblib.dump(model_pipeline, MODEL_PATH)
        
        update_training_status()
        
        return f"Training complete. Total Samples: {df.shape[0]}. Accuracy on test set: {accuracy:.4f}"

    except Exception as e:
        return f"Training failed unexpectedly: {str(e)}"

def load_model():
    """Loads the trained pipeline for prediction."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

def predict_message(message_text, model):
    """Uses the loaded model to predict the label of a new message."""
    if model is None:
        return "Model not trained.", 0.5
    
    cleaned_text = clean_text(message_text)
    
    prediction = model.predict([cleaned_text])[0]
    probability_spam = model.predict_proba([cleaned_text])[0][1]
    
    label = "spam" if prediction == 1 else "ham"
    return label, probability_spam