import requests
import json
import os
import numpy as np
import pandas as pd

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "<your-endpoint-url>")
API_KEY      = os.environ.get("API_KEY", "<your-api-key>")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def load_data(path):
    parquet = os.path.join(path, "data.parquet") if not path.endswith(".parquet") else path
    return pd.read_parquet(parquet)

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
    return df[use_cols].values.astype(np.float32)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_data", type=str, required=True)
    args = parser.parse_args()

    print("Loading deployment dataset...")
    df = load_data(args.deploy_data)
    df["label"] = (df["overall"] >= 4).astype(int)
    X = build_features(df)
    y_true = df["label"].values

    print(f"Sending {len(X)} samples to endpoint...")
    payload = {"data": X.tolist()}
    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload))

    result = response.json()
    if "error" in result:
        print("Error:", result["error"])
        return

    preds = np.array(result["predictions"])
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, preds)
    print(f"Deployment accuracy: {acc:.4f}")
    print(f"Predictions (first 10): {preds[:10].tolist()}")

if __name__ == "__main__":
    main()
