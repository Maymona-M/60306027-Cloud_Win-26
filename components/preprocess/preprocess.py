import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--output_data', type=str)
    args = parser.parse_args()

    # Load
    train_df = pd.read_csv(os.path.join(args.input_data, 'train_raw.csv'))
    print(f"Input shape: {train_df.shape}")

    # Compute RUL
    max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='engine_id')
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df.drop(columns=['max_cycle'], inplace=True)

    # Drop flat sensors
    flat_sensors = ['sensor_1', 'sensor_5', 'sensor_6',
                    'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    train_df.drop(columns=flat_sensors, inplace=True)

    # Normalize
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])

    print(f"Output shape: {train_df.shape}")
    print(f"RUL range: {train_df['RUL'].min()} to {train_df['RUL'].max()}")

    # Save
    os.makedirs(args.output_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_data, 'train_preprocessed.csv'), index=False)
    joblib.dump(scaler, os.path.join(args.output_data, 'scaler.pkl'))
    print("Preprocess complete.")

if __name__ == '__main__':
    main()