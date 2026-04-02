import os
import argparse
import pandas as pd
import mlflow
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessor


def main():
    # ----------------------------
    # ARGUMENTS
    # ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)

    args = parser.parse_args()

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    train_path = os.path.join(args.train_data, "train.csv")
    val_path = os.path.join(args.val_data, "val.csv")

    print(f"[INFO] Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"[INFO] Loading validation data from {val_path}")
    val_df = pd.read_csv(val_path)

    # ----------------------------
    # SPLIT FEATURES
    # ----------------------------
    def split(df):
        y = df["Survived"].copy()
        X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
        return X, y

    X_train, y_train = split(train_df)
    X_val, y_val = split(val_df)

    # ----------------------------
    # PREPROCESS + MODEL
    # ----------------------------
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    pre = build_preprocessor(numeric_features, categorical_features)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model)
    ])

    # ----------------------------
    # TRAIN
    # ----------------------------
    pipe.fit(X_train, y_train)

    # ----------------------------
    # EVALUATE
    # ----------------------------
    val_acc = pipe.score(X_val, y_val)
    print(f"[INFO] Validation Accuracy: {val_acc}")

    # ----------------------------
    # LOG METRICS (CRITICAL)
    # ----------------------------
    mlflow.log_metric("val_accuracy", val_acc)

    # ----------------------------
    # SAVE MODEL (CRITICAL)
    # ----------------------------
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(pipe, "outputs/model.pkl")

    print("[INFO] Model saved to outputs/model.pkl")


if __name__ == "__main__":
    main()
