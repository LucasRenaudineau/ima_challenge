import os
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from dotenv import load_dotenv

# Reuse the dataset builder and saver from feature_extraction.py
from features import build_feature_dataset, save_features

# Hyperparameters

CROP_PARAMS = (104, 104, 159)
THRESHOLDS  = (0.68, 0.92, 0.08, 0.10, 0.97)

FEATURES_CSV          = "outputs/features.csv"
IMPORTANCES_PNG       = "outputs/feature_importances.png"

TEST_SIZE             = 0.2
RANDOM_STATE          = 42
N_ESTIMATORS          = 100
N_ITER_SEARCH         = 20
CV_FOLDS              = 3

PARAM_DISTRIBUTIONS = {
    "max_depth":        [4, 6, 8, 10, 12],
    "min_child_weight": [1, 3, 5, 7],
    "subsample":        [0.5, 0.7, 0.9],
    "colsample_bynode": [0.5, 0.7, 0.9],
}

# Data loading

def load_features(features_csv):
    # Loads the already computed csv file with the features
    return pd.read_csv(features_csv)


def merge_with_labels(df_features, metadata_csv_path):
    df_labels = pd.read_csv(metadata_csv_path)
    df_labels = df_labels.rename(columns={"ID": "image_id"})
    df_merged = pd.merge(df_features, df_labels, on="image_id")
    return df_merged


def encode_labels(y):
    # Fits the label
    encoder    = LabelEncoder()
    y_encoded  = encoder.fit_transform(y)
    return y_encoded, encoder


def split_dataset(X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )


def compute_weights(y_train):
    return compute_sample_weight(class_weight="balanced", y=y_train)


def prepare_data(df_merged):
    # There are 3 steps here :
    # - Extract X and y from the merged DataFrame, encode labels,
    # - Split into train/test sets, and compute sample weights.
    # - Return X_train, X_test, y_train, y_test, encoder, sample_weights.
    X         = df_merged.drop(columns=["label", "image_id"], errors="ignore")
    y         = df_merged["label"]
    y_encoded, encoder = encode_labels(y)
    X_train, X_test, y_train, y_test = split_dataset(X, y_encoded)
    weights   = compute_weights(y_train)
    return X_train, X_test, y_train, y_test, encoder, weights

# Training

def build_base_model(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE):
    return xgb.XGBRFClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def run_hyperparameter_search(
    base_model,
    X_train,
    y_train,
    sample_weights,
    param_distributions=PARAM_DISTRIBUTIONS,
    n_iter=N_ITER_SEARCH,
    cv=CV_FOLDS,
    random_state=RANDOM_STATE,
):
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train, sample_weight=sample_weights)
    return search


def get_best_model(search):
    return search.best_estimator_


def train_model(X_train, y_train, sample_weights): # full pipeline
    print("\nTraining XGBoost Random Forest model...")
    base_model = build_base_model()

    print("Searching best hyperparameters...")
    search     = run_hyperparameter_search(base_model, X_train, y_train, sample_weights)
    best_model = get_best_model(search)

    print("\nBest parameters found:")
    print(search.best_params_)

    return best_model, search.best_params_

# Evaluation

def predict(model, X_test):
    return model.predict(X_test)


def compute_f1(y_test, y_pred):
    return f1_score(y_test, y_pred, average="macro")


def decode_and_report(encoder, y_test, y_pred):
    # Print a report
    y_test_decoded = encoder.inverse_transform(y_test)
    y_pred_decoded = encoder.inverse_transform(y_pred)
    print("\n=== Classification Report ===")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))


def evaluate_model(model, X_test, y_test, encoder):
    y_pred    = predict(model, X_test)
    macro_f1  = compute_f1(y_test, y_pred)
    print(f"\nF1-Score Macro: {macro_f1:.4f}")
    decode_and_report(encoder, y_test, y_pred)
    return y_pred

# Visualisation of feature importance

def compute_feature_importances(model, feature_names):
    return (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=True)
    )

def plot_feature_importances(importances, output_path=IMPORTANCES_PNG):
    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="purple")
    plt.title("Feature importances")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importances saved to {output_path}")

def show_feature_importances(model, X_train, output_path=IMPORTANCES_PNG):
    importances = compute_feature_importances(model, X_train.columns)
    plot_feature_importances(importances, output_path)

if __name__ == "__main__":
    load_dotenv()

    FOLDER_PATH = os.getenv("TRAIN_FOLDER_PATH")
    CSV_PATH = os.getenv("METADATA_CSV_PATH")

    if not FOLDER_PATH or not CSV_PATH:
        raise ValueError("ERROR: TRAIN_FOLDER_PATH or METADATA_CSV_PATH not defined in .env!")

    # Load or rebuild features
    if os.path.exists(FEATURES_CSV):
        print(f"Loading existing features from {FEATURES_CSV}")
        df_features = load_features(FEATURES_CSV)
    else:
        print("No features CSV found, rebuilding from images...")
        df_features = build_feature_dataset(FOLDER_PATH, CROP_PARAMS, THRESHOLDS)
        save_features(df_features, FEATURES_CSV)

    # Merge, prepare, train, evaluate
    df_merged = merge_with_labels(df_features, CSV_PATH)
    X_train, X_test, y_train, y_test, encoder, weights = prepare_data(df_merged)

    best_model, best_params = train_model(X_train, y_train, weights)

    evaluate_model(best_model, X_test, y_test, encoder)

    show_feature_importances(best_model, X_train)
