
import argparse
import sys
from pathlib import Path

import pandas as pd

from model import PhishingDetectionModel, DEFAULT_MODEL_PATH
from database import Database

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "sample_emails.csv"


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"label", "subject", "body"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
    df = df.dropna(subset=["label", "body"])
    df["subject"] = df["subject"].fillna("")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train the phishing detection model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH),
                         help="Path to labeled CSV dataset (columns: label, subject, body).")
    parser.add_argument("--algorithm", default="random_forest",
                         choices=["random_forest", "logistic_regression"],
                         help="Classifier algorithm to use.")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH),
                         help="Where to save the trained model.")
    parser.add_argument("--test-size", type=float, default=0.25,
                         help="Fraction of data held out for evaluation.")
    args = parser.parse_args()

    print(f"[*] Loading dataset from {args.data} ...")
    df = load_dataset(args.data)
    print(f"[*] Loaded {len(df)} labeled emails "
          f"({(df['label'] == 'phishing').sum()} phishing / "
          f"{(df['label'] == 'legitimate').sum()} legitimate)")

    print(f"[*] Training {args.algorithm} model ...")
    model = PhishingDetectionModel(algorithm=args.algorithm)
    metrics = model.train(
        subjects=df["subject"].tolist(),
        bodies=df["body"].tolist(),
        labels=df["label"].tolist(),
        test_size=args.test_size,
    )

    print("\n=== Evaluation Results ===")
    print(f"Model version : {metrics['model_version']}")
    print(f"Accuracy      : {metrics['accuracy']}")
    print(f"Precision     : {metrics['precision']}")
    print(f"Recall        : {metrics['recall']}")
    print(f"F1 Score      : {metrics['f1_score']}")
    print(f"Confusion Matrix (rows=actual, cols=predicted [legit, phishing]):")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    model.save(args.model_out)
    print(f"[*] Model saved to {args.model_out}")

    db = Database()
    db.add_model_metadata(
        model_version=metrics["model_version"],
        algorithm=metrics["algorithm"],
        accuracy=metrics["accuracy"],
        precision_score=metrics["precision"],
        recall_score=metrics["recall"],
        f1_score=metrics["f1_score"],
    )
    print("[*] Metrics recorded in database.")


if __name__ == "__main__":
    sys.exit(main())
