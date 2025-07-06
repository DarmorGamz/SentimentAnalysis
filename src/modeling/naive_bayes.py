from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os

def train_naive_bayes(news_df, labels_df, output_dir="models/naive_bayes"):
    """
    Train a Naive Bayes model for sentiment analysis.
    """
    # Merge news and labels on date
    data = pd.merge(news_df, labels_df, left_on="date", right_on="Date")
    X = data["cleaned_text"]
    y = data["sentiment"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "naive_bayes_model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    return model

if __name__ == "__main__":
    news_df = pd.read_csv("data/processed/news_data/processed_news.csv")
    labels_df = pd.read_csv("data/processed/sentiment_labels/sentiment_labels.csv")
    model = train_naive_bayes(news_df, labels_df)