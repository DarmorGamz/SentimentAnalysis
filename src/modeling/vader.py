# filename: vader.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from src.modeling.models import SentimentModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VaderModel(SentimentModel):
    """
    VADER model for sentiment analysis implementing the SentimentModel interface.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def train(self, X: pd.Series, y: pd.Series, output_dir: str) -> None:
        """
        VADER is rule-based and does not require training.
        """
        logger.info("VADER model does not require training.")
        os.makedirs(output_dir, exist_ok=True)

    def predict(self, X: pd.Series) -> pd.Series:
        """
        Predict sentiment labels for the given text data.
        """
        logger.info("Predicting with VADER model...")
        predictions = []
        for text in X:
            score = self.analyzer.polarity_scores(text)['compound']
            if score > 0.05:
                predictions.append("bullish")
            elif score < -0.05:
                predictions.append("bearish")
            else:
                predictions.append("neutral")
        return pd.Series(predictions, index=X.index)

    def evaluate(self, X: pd.Series, y: pd.Series, output_dir: str) -> dict:
        """
        Evaluate the model and save metrics and confusion matrix plot.
        """
        logger.info("Evaluating VADER model...")
        y_pred = self.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted")
        
        # Log metrics
        logger.info(f"VADER Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        # Save metrics to a file
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "vader_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("VADER Performance:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y, y_pred, labels=["bullish", "neutral", "bearish"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Bullish", "Neutral", "Bearish"], 
                    yticklabels=["Bullish", "Neutral", "Bearish"])
        plt.title("VADER Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, "vader_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {cm_path}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    news_df = pd.read_csv("data/processed/news_data/processed_news.csv")
    labels_df = pd.read_csv("data/processed/sentiment_labels/sentiment_labels.csv")
    
    data = pd.merge(news_df, labels_df, left_on=["date", "ticker"], right_on=["Date", "ticker"])
    data = data.dropna(subset=["cleaned_text"])
    
    X = data["cleaned_text"]
    y = data["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = VaderModel()
    model.train(X_train, y_train, "models/vader")
    metrics = model.evaluate(X_test, y_test, "models/vader")