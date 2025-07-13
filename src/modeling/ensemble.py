# filename: ensemble.py
from src.modeling.bert import BertModel
from src.modeling.vader import VaderModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from src.modeling.models import SentimentModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EnsembleModel(SentimentModel):
    """
    Ensemble model combining BERT and VADER with hyperparameter tuning.
    """
    def __init__(self):
        self.bert = BertModel()
        self.vader = VaderModel()
        self.weights = [0.7, 0.3]  # Default weights: BERT 70%, VADER 30%

    def train(self, X: pd.Series, y: pd.Series, output_dir: str) -> None:
        """
        Train the ensemble: train BERT, tune VADER threshold and weights on validation set.
        """
        if len(X) < 20:
            logger.error(f"Dataset too small for ensemble training: {len(X)} samples")
            raise ValueError(f"Dataset too small for ensemble training: {len(X)} samples")
        
        # Split into sub-train and validation for tuning
        X_sub, X_val, y_sub, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train BERT on sub-train
        bert_dir = os.path.join(output_dir, "bert")
        self.bert.train(X_sub, y_sub, bert_dir)
        
        # Tune hyperparameters on validation
        logger.info("Tuning ensemble hyperparameters...")
        thresh_values = np.arange(0.0, 0.35, 0.05)
        weight_values = np.arange(0.5, 1.0, 0.1)
        best_acc = 0
        best_thresh = 0.05
        best_weight_bert = 0.7
        
        bert_probs = self.bert.predict_proba(X_val)
        
        for thresh in thresh_values:
            self.vader.pos_thresh = thresh
            self.vader.neg_thresh = -thresh
            vader_probs = self.vader.predict_proba(X_val)
            for w_bert in weight_values:
                w_vader = 1 - w_bert
                ens_probs = w_bert * bert_probs + w_vader * vader_probs
                y_pred = ens_probs.idxmax(axis=1)
                acc = accuracy_score(y_val, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
                    best_weight_bert = w_bert
        
        self.vader.pos_thresh = best_thresh
        self.vader.neg_thresh = -best_thresh
        self.weights = [best_weight_bert, 1 - best_weight_bert]
        
        logger.info(f"Best VADER threshold: {best_thresh}, best BERT weight: {best_weight_bert}, val accuracy: {best_acc:.4f}")
        
        # Save parameters
        os.makedirs(output_dir, exist_ok=True)
        params_path = os.path.join(output_dir, "ensemble_params.json")
        with open(params_path, "w") as f:
            json.dump({"pos_thresh": best_thresh, "weight_bert": best_weight_bert}, f)
        logger.info(f"Saved ensemble parameters to {params_path}")

    def predict_proba(self, X: pd.Series) -> pd.DataFrame:
        """
        Predict ensemble sentiment probabilities.
        """
        bert_probs = self.bert.predict_proba(X)
        vader_probs = self.vader.predict_proba(X)
        return self.weights[0] * bert_probs + self.weights[1] * vader_probs

    def predict(self, X: pd.Series) -> pd.Series:
        """
        Predict ensemble sentiment labels.
        """
        logger.info("Predicting with Ensemble model...")
        return self.predict_proba(X).idxmax(axis=1)

    def evaluate(self, X: pd.Series, y: pd.Series, output_dir: str) -> dict:
        """
        Evaluate the ensemble model and save metrics and confusion matrix plot.
        """
        logger.info("Evaluating Ensemble model...")
        y_pred = self.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted")
        
        # Log metrics
        logger.info(f"Ensemble Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        # Save metrics to a file
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "ensemble_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("Ensemble Performance:\n")
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
        plt.title("Ensemble Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, "ensemble_confusion_matrix.png")
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
    
    model = EnsembleModel()
    model.train(X_train, y_train, "models/ensemble")
    metrics = model.evaluate(X_test, y_test, "models/ensemble")