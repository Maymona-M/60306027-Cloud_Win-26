import argparse
import os
import time
import pandas as pd

def read_parquet(path):
    parquet_path = os.path.join(path, "data.parquet")
    return pd.read_parquet(parquet_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length",    type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf",     type=str, required=True)
    parser.add_argument("--sbert",     type=str, required=True)
    parser.add_argument("--out",       type=str, required=True)
    return parser.parse_args()

def main():
    t_start = time.time()
    args = parse_args()

    KEY = ["asin", "reviewerID"]

    length_df    = read_parquet(args.length).drop_duplicates(subset=KEY)
    sentiment_df = read_parquet(args.sentiment).drop_duplicates(subset=KEY)
    tfidf_df     = read_parquet(args.tfidf).drop_duplicates(subset=KEY)
    sbert_df     = read_parquet(args.sbert).drop_duplicates(subset=KEY)

    print(f"length:    {length_df.shape}")
    print(f"sentiment: {sentiment_df.shape}")
    print(f"tfidf:     {tfidf_df.shape}")
    print(f"sbert:     {sbert_df.shape}")

    merged = length_df.merge(sentiment_df, on=KEY, how="inner")
    merged = merged.merge(tfidf_df,        on=KEY, how="inner")
    merged = merged.merge(sbert_df,        on=KEY, how="inner")

    print(f"Merged shape: {merged.shape}")

    os.makedirs(args.out, exist_ok=True)
    merged.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"Written to: {args.out}")

    elapsed = time.time() - t_start
    print(f"[PIPELINE TIMING] merge_all completed in {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()