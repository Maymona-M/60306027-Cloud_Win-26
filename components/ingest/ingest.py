import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str)
    parser.add_argument('--output_data', type=str)
    args = parser.parse_args()

    # Column names
    cols = ['engine_id', 'cycle',
            'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
           [f'sensor_{i}' for i in range(1, 22)]

    # Load raw data
    train_df = pd.read_csv(
        os.path.join(args.raw_data, 'train_FD001.txt'),
        sep=r'\s+', header=None, names=cols
    )

    print(f"Loaded shape: {train_df.shape}")
    print(f"Engines: {train_df['engine_id'].nunique()}")

    # Save
    os.makedirs(args.output_data, exist_ok=True)
    train_df.to_csv(
        os.path.join(args.output_data, 'train_raw.csv'), index=False
    )
    print("Ingest complete.")

if __name__ == '__main__':
    main()