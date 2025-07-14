# filename: src/modeling/vader.py
import os
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import nltk
nltk.download('vader_lexicon', quiet=True)
from src.modeling.models import SentimentModel
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VaderModel(SentimentModel):
    """
    VADER model for sentiment classification.
    """
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.pos_thresh = 0.05
        self.neg_thresh = -0.05

    def train(self, X: pd.Series, y: pd.Series, output_dir: str) -> None:
        # VADER is rule-based, no training
        pass

    def predict(self, X: pd.Series) -> pd.Series:
        def get_sentiment(text):
            score = self.sia.polarity_scores(text)['compound']
            if score > self.pos_thresh:
                return 'bullish'
            elif score < self.neg_thresh:
                return 'bearish'
            else:
                return 'neutral'
        return X.apply(get_sentiment)

    def predict_proba(self, X: pd.Series) -> pd.DataFrame:
        probs = []
        for text in X:
            score = self.sia.polarity_scores(text)['compound']
            bull = max((score - self.neg_thresh) / (self.pos_thresh - self.neg_thresh + 1), 0) if score > 0 else 0
            bear = max((self.pos_thresh - score) / (self.pos_thresh - self.neg_thresh + 1), 0) if score < 0 else 0
            neut = 1 - bull - bear
            probs.append([bull, neut, bear])
        return pd.DataFrame(probs, columns=['bullish', 'neutral', 'bearish'])

    def evaluate(self, X: pd.Series, y: pd.Series, output_dir: str) -> dict:
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'vader_metrics.txt'), 'w') as f:
            json.dump(metrics, f)

        cm = confusion_matrix(y, y_pred, labels=['bullish', 'neutral', 'bearish'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bullish', 'Neutral', 'Bearish'],
                    yticklabels=['Bullish', 'Neutral', 'Bearish'])
        plt.title('VADER Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'vader_confusion_matrix.png'))
        plt.close()

        return metrics

    def load(self, output_dir: str) -> None:
        # No load needed for VADER
        pass