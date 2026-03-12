import argparse
import pandas as pd
import time
import os
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--output_data', type=str)
    args = parser.parse_args()

    # Load
    train_df = pd.read_csv(os.path.join(args.input_data, 'train_preprocessed.csv'))
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
    print(f"Input shape: {train_df.shape}")

    # Melt to long format
    train_long = train_df.melt(
        id_vars=['engine_id', 'cycle'],
        value_vars=sensor_cols,
        var_name='sensor',
        value_name='value'
    )
    train_long['id'] = train_long['engine_id'].astype(str) + '_' + train_long['sensor']
    print(f"Long format shape: {train_long.shape}")

    # Extract features
    print("Starting tsfresh extraction...")
    start = time.time()

    features = extract_features(
        train_long,
        column_id='id',
        column_sort='cycle',
        column_value='value',
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=0,
        disable_progressbar=False
    )

    elapsed = time.time() - start
    print(f"Extraction done in {elapsed:.1f} seconds")

    # Reshape to per-engine
    features.index = features.index.astype(str)
    features['engine_id'] = features.index.str.split('_').str[0].astype(int)
    features_per_engine = features.groupby('engine_id').mean()

    # Attach RUL labels
    rul_labels = train_df.groupby('engine_id')['RUL'].max().reset_index()
    rul_labels.columns = ['engine_id', 'RUL']
    features_per_engine = features_per_engine.reset_index()
    dataset = features_per_engine.merge(rul_labels, on='engine_id')

    print(f"Final dataset shape: {dataset.shape}")

    # Save
    os.makedirs(args.output_data, exist_ok=True)
    dataset.to_csv(os.path.join(args.output_data, 'features_with_rul.csv'), index=False)

    # Save runtime
    import json
    with open(os.path.join(args.output_data, 'extraction_time.json'), 'w') as f:
        json.dump({'extraction_seconds': round(elapsed, 1)}, f)

    print("Feature extraction complete.")

if __name__ == '__main__':
    main()