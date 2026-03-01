import argparse
import os
import re
import pandas as pd

# Compiled once at module level for performance
URL_PATTERN   = re.compile(r"https?://\S+|www\.\S+")
NUM_PATTERN   = re.compile(r"\b\d+\b")
PUNCT_PATTERN = re.compile(r"[^a-z\s]")
SPACE_PATTERN = re.compile(r"\s+")

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Step 1: lowercase
    text = text.lower()
    # Step 2: remove URLs
    text = URL_PATTERN.sub(" ", text)
    # Step 3: replace numbers
    text = NUM_PATTERN.sub(" ", text)
    # Step 4: remove punctuation
    text = PUNCT_PATTERN.sub(" ", text)
    # Step 5: collapse whitespace
    text = SPACE_PATTERN.sub(" ", text).strip()
    return text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out",  type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    df = pd.read_parquet(args.data)
    print(f"Rows loaded: {len(df)}")

    df["reviewText"] = df["reviewText"].apply(normalize_text)

    before = len(df)
    df = df[df["reviewText"].str.len() >= 10].reset_index(drop=True)
    print(f"Rows after filtering short reviews: {len(df)} (dropped {before - len(df)})")

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"Written to: {args.out}")

if __name__ == "__main__":
    main()