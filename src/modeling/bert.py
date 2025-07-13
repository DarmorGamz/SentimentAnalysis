# filename: bert.py
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
import torch
import numpy as np
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

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class SentimentDataset(Dataset):
    """
    Custom dataset for sentiment analysis with BERT.
    """
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        label_map = {"bullish": 0, "neutral": 1, "bearish": 2}
        self.labels = [label_map[l] for l in labels]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SaveTokenizerCallback(TrainerCallback):
    """
    Callback to save tokenizer during checkpointing.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved tokenizer to {checkpoint_dir}")

class BertModel(SentimentModel):
    """
    FinBERT model for sentiment analysis implementing the SentimentModel interface.
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3).to(device)

    def train(self, X: pd.Series, y: pd.Series, output_dir: str) -> None:
        """
        Train the FinBERT model on the given data, resuming from last checkpoint if available.
        """
        if len(X) < 10:
            logger.error(f"Dataset too small: {len(X)} samples")
            raise ValueError(f"Dataset too small: {len(X)} samples")
        
        logger.info("Training FinBERT model...")
        train_dataset = SentimentDataset(X, y, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            save_steps=500,
            logging_steps=100,
            eval_strategy="no",
            load_best_model_at_end=False
        )
        
        # Check for last checkpoint
        last_checkpoint = None
        if os.path.isdir(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split('-')[1]), reverse=True)
                last_checkpoint = os.path.join(output_dir, checkpoints_sorted[0])
        
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            self.model = BertForSequenceClassification.from_pretrained(last_checkpoint, num_labels=3).to(device)
            vocab_path = os.path.join(last_checkpoint, 'vocab.txt')
            if os.path.exists(vocab_path):
                self.tokenizer = BertTokenizer.from_pretrained(last_checkpoint)
            else:
                self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
                logger.warning(f"Tokenizer not found in checkpoint, using pretrained 'ProsusAI/finbert'")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        trainer.add_callback(SaveTokenizerCallback(self.tokenizer))
        
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved model and tokenizer to {output_dir}")

    def predict_proba(self, X: pd.Series) -> pd.DataFrame:
        """
        Predict sentiment probabilities for the given text data.
        """
        logger.info("Predicting probabilities with FinBERT model...")
        texts = list(X)
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return pd.DataFrame(probs, columns=["bullish", "neutral", "bearish"], index=X.index)

    def predict(self, X: pd.Series) -> pd.Series:
        """
        Predict sentiment labels for the given text data.
        """
        return self.predict_proba(X).idxmax(axis=1)

    def evaluate(self, X: pd.Series, y: pd.Series, output_dir: str) -> dict:
        """
        Evaluate the model and save metrics and confusion matrix plot.
        """
        logger.info("Evaluating FinBERT model...")
        y_pred = self.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted")
        
        # Log metrics
        logger.info(f"FinBERT Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        # Save metrics to a file
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "finbert_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("FinBERT Performance:\n")
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
        plt.title("FinBERT Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, "finbert_confusion_matrix.png")
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
    
    model = BertModel()
    model.train(X_train, y_train, "models/bert")
    metrics = model.evaluate(X_test, y_test, "models/bert")