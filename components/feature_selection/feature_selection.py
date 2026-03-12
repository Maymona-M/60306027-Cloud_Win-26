import argparse
import pandas as pd
import numpy as np
import json
import random
import time
import os
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from deap import base, creator, tools, algorithms

def filter_features(X, y):
    # Step 1: Variance threshold
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    surviving = X.columns[selector.get_support()]
    X_var_df = pd.DataFrame(X_var, columns=surviving)
    print(f"After variance filter: {X_var_df.shape[1]}")

    # Step 2: Correlation filter
    corr_matrix = X_var_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_corr_df = X_var_df.drop(columns=to_drop)
    print(f"After correlation filter: {X_corr_df.shape[1]}")

    # Fix NaNs
    X_corr_df = X_corr_df.fillna(X_corr_df.median())

    # Step 3: Mutual information top 50
    mi_scores = mutual_info_regression(X_corr_df, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_corr_df.columns).sort_values(ascending=False)
    top_50 = mi_series.head(50).index
    X_filtered = X_corr_df[top_50]
    print(f"After mutual info filter: {X_filtered.shape[1]}")

    return X_filtered

def run_ga(X_filtered, y):
    N_FEATURES = X_filtered.shape[1]

    # Avoid re-creating if already exists
    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    def evaluate(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected) < 2:
            return (9999,)
        X_sel = X_filtered.iloc[:, selected]
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        scores = cross_val_score(model, X_sel, y,
                                 cv=5,
                                 scoring='neg_root_mean_squared_error')
        rmse = -scores.mean()
        penalty = 0.5 * len(selected)
        return (rmse + penalty,)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, N_FEATURES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(42)
    pop = toolbox.population(n=30)

    print("Running Genetic Algorithm...")
    start = time.time()
    result, _ = algorithms.eaSimple(pop, toolbox,
                                    cxpb=0.7, mutpb=0.2,
                                    ngen=20, verbose=True)
    ga_time = time.time() - start
    print(f"GA done in {ga_time:.1f} seconds")

    best = tools.selBest(result, k=1)[0]
    selected_indices = [i for i, bit in enumerate(best) if bit == 1]
    selected_features = X_filtered.columns[selected_indices].tolist()

    return selected_features, ga_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--output_data', type=str)
    args = parser.parse_args()

    # Load
    dataset = pd.read_csv(os.path.join(args.input_data, 'features_with_rul.csv'))
    X = dataset.drop(columns=['engine_id', 'RUL'])
    y = dataset['RUL']
    print(f"Input shape: {X.shape}")

    # Stage 1: Filter
    X_filtered = filter_features(X, y)

    # Stage 2: GA
    selected_features, ga_time = run_ga(X_filtered, y)
    print(f"Selected {len(selected_features)} features")

    # Save
    os.makedirs(args.output_data, exist_ok=True)

    X_final = X_filtered[selected_features]
    X_final.to_csv(os.path.join(args.output_data, 'features_final.csv'), index=False)
    y.to_csv(os.path.join(args.output_data, 'rul_labels.csv'), index=False)

    with open(os.path.join(args.output_data, 'selected_features.json'), 'w') as f:
        json.dump(selected_features, f, indent=2)

    with open(os.path.join(args.output_data, 'ga_runtime.json'), 'w') as f:
        json.dump({'ga_seconds': round(ga_time, 1),
                   'n_selected': len(selected_features)}, f)

    print("Feature selection complete.")

if __name__ == '__main__':
    main()