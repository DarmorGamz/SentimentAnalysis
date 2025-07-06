import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model(y_true, y_pred, model_name="naive_bayes", output_dir="models/naive_bayes"):
    """
    Evaluate model performance and save confusion matrix plot.
    
    Parameters:
    - y_true: True labels (bullish, neutral, bearish)
    - y_pred: Predicted labels
    - model_name: Name of the model (for output file naming)
    - output_dir: Directory to save results
    """
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    
    # Print metrics
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save metrics to a file
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"{model_name} Performance:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    print(f"Saved metrics to {metrics_path}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["bullish", "neutral", "bearish"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Bullish", "Neutral", "Bearish"], 
                yticklabels=["Bullish", "Neutral", "Bearish"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    # Example usage with Naive Bayes predictions
    news_df = pd.read_csv("data/processed/news_data/processed_news.csv")
    labels_df = pd.read_csv("data/processed/sentiment_labels/sentiment_labels.csv")
    data = pd.merge(news_df, labels_df, left_on="date", right_on="Date")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    
    X = data["cleaned_text"]
    y = data["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    evaluate_model(y_test, y_pred, model_name="Naive Bayes")