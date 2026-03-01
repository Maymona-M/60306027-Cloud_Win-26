import argparse
import os
import gc
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, required=True)
    parser.add_argument("--out",        type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=10000)
    return parser.parse_args()

def main():
    args = parse_args()

    parquet_path = os.path.join(args.data, "data.parquet")
    df = pd.read_parquet(parquet_path)
    print(f"Rows loaded: {len(df)}")

    model = SentenceTransformer(args.model_name)
    texts = df["reviewText"].fillna("").tolist()

    all_embeddings = []
    total = len(texts)

    for start in range(0, total, args.chunk_size):
        end = min(start + args.chunk_size, total)
        chunk = texts[start:end]
        print(f"Encoding chunk {start}-{end} of {total}...")
        emb = model.encode(
            chunk,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(emb.astype(np.float32))
        del emb
        gc.collect()

    embeddings = np.vstack(all_embeddings)
    print(f"Embedding shape: {embeddings.shape}")

    cols   = [f"bert_emb_{i}" for i in range(embeddings.shape[1])]
    out_df = pd.DataFrame(embeddings.astype(np.float32), columns=cols)
    out_df.insert(0, "reviewerID", df["reviewerID"].values)
    out_df.insert(0, "asin",       df["asin"].values)

    del embeddings
    gc.collect()

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"Written to: {args.out}")

if __name__ == "__main__":
    main()