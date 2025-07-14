# filename: src/modeling/financial_bert.py
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.modeling.models import SentimentModel
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FinancialBertModel(SentimentModel):
    """
    FinancialBERT model for sentiment classification.
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
        self.label_map = {'bullish': 0, 'neutral': 1, 'bearish': 2}
        self.inv_map = {0: 'bullish', 1: 'neutral', 2: 'bearish'}

    def _prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=512)
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }
        return Dataset.from_dict(dataset_dict)

    def train(self, X: pd.Series, y: pd.Series, output_dir: str) -> None:
        logger.info("Training FinancialBERT on Financial PhraseBank...")
        df_gen = pd.read_csv('./data/training/financial_phrasebank.csv')
        df_gen['Sentiment'] = df_gen['Sentiment'].map({'negative': 2, 'positive': 0, 'neutral': 1})
        df_gen.rename(columns={'Sentence': 'text', 'Sentiment': 'label'}, inplace=True)
        train_df, val_df = train_test_split(df_gen, test_size=0.2, random_state=42, stratify=df_gen['label'])
        train_ds = self._prepare_dataset(train_df)
        val_ds = self._prepare_dataset(val_df)
        train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = np.argmax(pred.predictions, axis=-1)
            acc = accuracy_score(labels, preds)
            return {'accuracy': acc}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, X: pd.Series) -> pd.Series:
        dataset = self._prepare_dataset(pd.DataFrame({'text': X, 'label': [0]*len(X)}))  # Dummy labels
        dataset.set_format('torch')
        trainer = Trainer(model=self.model)
        preds = np.argmax(trainer.predict(dataset).predictions, axis=-1)
        return pd.Series([self.inv_map[p] for p in preds])

    def predict_proba(self, X: pd.Series) -> pd.DataFrame:
        dataset = self._prepare_dataset(pd.DataFrame({'text': X, 'label': [0]*len(X)}))  # Dummy labels
        dataset.set_format('torch')
        trainer = Trainer(model=self.model)
        logits = trainer.predict(dataset).predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        cols = [self.inv_map[i] for i in range(3)]
        return pd.DataFrame(probs, columns=cols)

    def evaluate(self, X: pd.Series, y: pd.Series, output_dir: str) -> dict:
        logger.info("Evaluating FinancialBERT on Financial PhraseBank test split...")
        df_gen = pd.read_csv('./data/training/financial_phrasebank.csv')
        df_gen['Sentiment'] = df_gen['Sentiment'].map({'negative': 2, 'positive': 0, 'neutral': 1})
        df_gen.rename(columns={'Sentence': 'text', 'Sentiment': 'label'}, inplace=True)
        _, val_df = train_test_split(df_gen, test_size=0.2, random_state=42, stratify=df_gen['label'])
        val_ds = self._prepare_dataset(val_df)
        val_ds.set_format('torch')
        trainer = Trainer(model=self.model)
        pred = trainer.predict(val_ds)
        y_pred = np.argmax(pred.predictions, axis=-1)
        y_true = [t.item() for t in val_ds['labels']]
        y_pred_labels = [self.inv_map[p] for p in y_pred]
        y_true_labels = [self.inv_map[t] for t in y_true]
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'financialbert_metrics.txt'), 'w') as f:
            f.write(str(metrics))

        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=['bullish', 'neutral', 'bearish'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bullish', 'Neutral', 'Bearish'],
                    yticklabels=['Bullish', 'Neutral', 'Bearish'])
        plt.title('FinancialBERT Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'financialbert_confusion_matrix.png'))
        plt.close()

        return metrics

    def load(self, output_dir: str) -> None:
        self.model = BertForSequenceClassification.from_pretrained(output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(output_dir)