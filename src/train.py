import argparse
import os
import time
import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data",   type=str, required=True)
    parser.add_argument("--test_data",  type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--C",          type=float, default=1.0)
    parser.add_argument("--max_iter",   type=int,   default=1000)
    return parser.parse_args()

def load_data(path):
    parquet = os.path.join(path, "data.parquet") if not path.endswith(".parquet") else path
    if not os.path.exists(parquet):
        raise FileNotFoundError(f"Not found: {parquet}")
    return pd.read_parquet(parquet)

def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing.")
    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

def build_features(df):
    bert_cols      = [c for c in df.columns if c.startswith("bert_emb_")]
    tfidf_cols     = [c for c in df.columns if c.startswith("tfidf_")]
    sentiment_cols = [c for c in df.columns if c in [
        "sentiment_positive", "sentiment_negative", "sentiment_neutral", "compound"
    ]]
    length_cols    = [c for c in df.columns if c in [
        "review_length", "word_count", "avg_word_length", "sentence_count"
    ]]
    use_cols = bert_cols + tfidf_cols + sentiment_cols + length_cols
    if not use_cols:
        raise RuntimeError("No feature columns found.")
    return df[use_cols].values.astype(np.float32)

def evaluate(model, X, y, split):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc  = accuracy_score(y, preds)
    auc  = roc_auc_score(y, proba)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    mlflow.log_metric(f"{split}_accuracy",  acc)
    mlflow.log_metric(f"{split}_auc",       auc)
    mlflow.log_metric(f"{split}_precision", prec)
    mlflow.log_metric(f"{split}_recall",    rec)
    mlflow.log_metric(f"{split}_f1",        f1)
    print(f"{split}: acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}")

def main():
    args = parse_args()
    start_time = time.time()

    mlflow.start_run()
    mlflow.log_param("C", args.C)
    mlflow.log_param("max_iter", args.max_iter)

    print("Loading data...")
    train_df = create_labels(load_data(args.train_data))
    val_df   = create_labels(load_data(args.val_data))
    test_df  = create_labels(load_data(args.test_data))

    print("Building features...")
    X_train = build_features(train_df); y_train = train_df["label"]
    X_val   = build_features(val_df);   y_val   = val_df["label"]
    X_test  = build_features(test_df);  y_test  = test_df["label"]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("Training model...")
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="saga",
                               n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val,   y_val,   "val")
    evaluate(model, X_test,  y_test,  "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"Done in {runtime:.1f}s")
    mlflow.end_run()

if __name__ == "__main__":
    main()
