import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",        type=str, required=True)
    parser.add_argument("--val",          type=str, required=True)
    parser.add_argument("--test",         type=str, required=True)
    parser.add_argument("--train_out",    type=str, required=True)
    parser.add_argument("--val_out",      type=str, required=True)
    parser.add_argument("--test_out",     type=str, required=True)
    parser.add_argument("--max_features", type=int, default=5000)
    return parser.parse_args()

def transform_and_save(vectorizer, df, out_path, name):
    matrix = vectorizer.transform(df["reviewText"].fillna("").tolist())
    cols   = [f"tfidf_{t}" for t in vectorizer.get_feature_names_out()]
    out_df = pd.DataFrame(matrix.toarray().astype(np.float32), columns=cols)
    out_df.insert(0, "reviewerID", df["reviewerID"].values)
    out_df.insert(0, "asin",       df["asin"].values)
    os.makedirs(out_path, exist_ok=True)
    out_df.to_parquet(os.path.join(out_path, "data.parquet"), index=False)
    print(f"{name}: {out_df.shape} -> {out_path}")

def main():
    args = parse_args()

    train_df = pd.read_parquet(args.train)
    val_df   = pd.read_parquet(args.val)
    test_df  = pd.read_parquet(args.test)

    # Fit ONLY on training data to prevent leakage
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    vectorizer.fit(train_df["reviewText"].fillna("").tolist())
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    transform_and_save(vectorizer, train_df, args.train_out, "train")
    transform_and_save(vectorizer, val_df,   args.val_out,   "val")
    transform_and_save(vectorizer, test_df,  args.test_out,  "test")

    joblib.dump(vectorizer, os.path.join(args.train_out, "tfidf_vectorizer.joblib"))
    print("Vectorizer saved.")

if __name__ == "__main__":
    main()