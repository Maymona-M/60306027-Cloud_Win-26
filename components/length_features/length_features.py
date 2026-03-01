import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out",  type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)

    # Number of whitespace-separated tokens
    df["review_length_words"] = df["reviewText"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )
    # Number of characters
    df["review_length_chars"] = df["reviewText"].apply(
        lambda x: len(str(x)) if isinstance(x, str) else 0
    )

    print(df[["review_length_words", "review_length_chars"]].describe())

    out_df = df[["asin", "reviewerID", "review_length_words", "review_length_chars"]]
    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"Written to: {args.out}")

if __name__ == "__main__":
    main()