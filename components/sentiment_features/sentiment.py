import argparse
import os
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out",  type=str, required=True)
    return parser.parse_args()

def score(text, sia):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"sentiment_pos": 0.0, "sentiment_neg": 0.0,
                "sentiment_neu": 1.0, "sentiment_compound": 0.0}
    s = sia.polarity_scores(text)
    return {"sentiment_pos": s["pos"], "sentiment_neg": s["neg"],
            "sentiment_neu": s["neu"], "sentiment_compound": s["compound"]}

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    print(f"Rows loaded: {len(df)}")

    sia = SentimentIntensityAnalyzer()
    scores = df["reviewText"].apply(lambda t: score(t, sia))
    scores_df = pd.DataFrame(scores.tolist())
    scores_df["asin"]       = df["asin"].values
    scores_df["reviewerID"] = df["reviewerID"].values

    out_df = scores_df[["asin", "reviewerID",
                         "sentiment_pos", "sentiment_neg",
                         "sentiment_neu", "sentiment_compound"]]

    print(out_df[["sentiment_pos", "sentiment_neg",
                  "sentiment_neu", "sentiment_compound"]].describe())

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"Written to: {args.out}")

if __name__ == "__main__":
    main()