# filename: src/modeling/bert.py
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.modeling.models import SentimentModel
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BertModel(SentimentModel):
    """
    BERT model for sentiment classification.
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.label_map = {'bullish': 0, 'neutral': 1, 'bearish': 2}
        self.inv_map = {v: k for k, v in self.label_map.items()}

    def _prepare_dataset(self, X: pd.Series, y: pd.Series = None) -> Dataset:
        texts = X.tolist()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512)
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        if y is not None:
            labels = [self.label_map[label] for label in y]
            dataset_dict['labels'] = labels
        return Dataset.from_dict(dataset_dict)

    def train(self, X: pd.Series, y: pd.Series, output_dir: str) -> None:
        # Pre-train on general sentiment data
        logger.info("Pre-training BERT on Financial PhraseBank...")
        from datasets import load_dataset
        ds = load_dataset("financial_phrasebank", "sentences_allagree")
        def map_label(example):
            lbl = example['label']
            if lbl == 0:  # negative
                return {'label': 2}  # bearish
            elif lbl == 1:  # positive
                return {'label': 0} # bullish
            else:
                return {'label': 1} # neutral
        ds = ds.map(map_label)
        train_ds = ds['train'].rename_column('sentence', 'text')
        def tokenize(batch):
            return self.tokenizer(batch['text'], truncation=True, padding=True, max_length=512)
        train_ds = train_ds.map(tokenize, batched=True)
        train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args_gen = TrainingArguments(
            output_dir='./results_general',
            num_train_epochs=1,  # Short for pre-train
            per_device_train_batch_size=8,
            warmup_steps=0,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer_gen = Trainer(
            model=self.model,
            args=training_args_gen,
            train_dataset=train_ds,
        )
        trainer_gen.train()

        # Fine-tune on specific stock data
        logger.info("Fine-tuning BERT on specific stock data...")
        specific_ds = self._prepare_dataset(X, y)
        specific_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            evaluation_strategy="epoch",
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
            train_dataset=specific_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, X: pd.Series) -> pd.Series:
        dataset = self._prepare_dataset(X)
        dataset.set_format('torch')
        trainer = Trainer(model=self.model)
        preds = np.argmax(trainer.predict(dataset).predictions, axis=-1)
        return pd.Series([self.inv_map[p] for p in preds])

    def predict_proba(self, X: pd.Series) -> pd.DataFrame:
        dataset = self._prepare_dataset(X)
        dataset.set_format('torch')
        trainer = Trainer(model=self.model)
        logits = trainer.predict(dataset).predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        cols = [self.inv_map[i] for i in range(3)]
        return pd.DataFrame(probs, columns=cols)

    def evaluate(self, X: pd.Series, y: pd.Series, output_dir: str) -> dict:
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'bert_metrics.txt'), 'w') as f:
            f.write(str(metrics))

        cm = confusion_matrix(y, y_pred, labels=['bullish', 'neutral', 'bearish'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bullish', 'Neutral', 'Bearish'],
                    yticklabels=['Bullish', 'Neutral', 'Bearish'])
        plt.title('BERT Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'bert_confusion_matrix.png'))
        plt.close()

        return metrics

    def load(self, output_dir: str) -> None:
        self.model = BertForSequenceClassification.from_pretrained(output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(output_dir)