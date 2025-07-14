# filename: src/modeling/naive_bayes.py
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import load_dataset
import joblib
from src.modeling.models import SentimentModel
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class NaiveBayesModel(SentimentModel):
    """
    Naive Bayes model for sentiment classification.
    """
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self, X: pd.Series, y: pd.Series, output_dir: str) -> None:
        # Pre-train on general sentiment data
        logger.info("Pre-training Naive Bayes on Financial PhraseBank...")
        ds = load_dataset("financial_phrasebank", "sentences_allagree")
        df_gen = pd.DataFrame(ds['train'])
        df_gen['label'] = df_gen['label'].map({0: 'bearish', 1: 'bullish', 2: 'neutral'})
        X_gen = df_gen['sentence']
        y_gen = df_gen['label']
        X_gen_vect = self.vectorizer.fit_transform(X_gen)
        self.model.partial_fit(X_gen_vect, y_gen, classes=['bullish', 'neutral', 'bearish'])

        # Train on specific stock data
        logger.info("Training Naive Bayes on specific stock data...")
        X_vect = self.vectorizer.transform(X)
        self.model.partial_fit(X_vect, y)

        # Save model
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))
        joblib.dump(self.model, os.path.join(output_dir, 'model.pkl'))

    def predict(self, X: pd.Series) -> pd.Series:
        X_vect = self.vectorizer.transform(X)
        return pd.Series(self.model.predict(X_vect))

    def evaluate(self, X: pd.Series, y: pd.Series, output_dir: str) -> dict:
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

        # Save metrics
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
            json.dump(metrics, f)

        # Confusion matrix plot
        cm = confusion_matrix(y, y_pred, labels=['bullish', 'neutral', 'bearish'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bullish', 'Neutral', 'Bearish'],
                    yticklabels=['Bullish', 'Neutral', 'Bearish'])
        plt.title('Naive Bayes Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        return metrics

    def load(self, output_dir: str) -> None:
        self.vectorizer = joblib.load(os.path.join(output_dir, 'vectorizer.pkl'))
        self.model = joblib.load(os.path.join(output_dir, 'model.pkl'))