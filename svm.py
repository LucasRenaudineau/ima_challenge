import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
#from features import load_features

FEATURES_CSV  = "./outputs/features.csv"
METADATA_CSV  = "./IMA205-challenge/train_metadata.csv"
OUTPUT_DIR    = "./outputs"

def load_features(input_csv: str = "outputs/features.csv") -> pd.DataFrame:
    # Loads a previously saved feature CSV from disk
    return pd.read_csv(input_csv)

# Converts raw seconds
def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


# Merges a features DataFrame with a metadata CSV using normalised image ID keys
def merge_with_labels(df_features, metadata_csv_path):
    print("\n" + "=" * 60)
    print("STEP 1 - Loading and merging data")
    print("=" * 60)

    t0 = time.time()
    df_labels = pd.read_csv(metadata_csv_path)
    print(f" Labels loaded - {len(df_labels):,} rows [{fmt_time(time.time()-t0)}]")

    # Auto-detect which column in the features DataFrame holds the image ID, this is done because across my multiple tries it can be "id" or "image_id"...
    id_candidates = [c for c in df_features.columns if c.lower() in ("id", "image_id")]
    if not id_candidates:
        raise ValueError(
            f"Could not find an ID column in features.\n"
            f"Available columns: {list(df_features.columns)}"
        )
    feat_id_col = id_candidates[0]
    print(f" Using '{feat_id_col}' as key in features, 'ID' as key in labels ...")

    df_features["_key"] = df_features[feat_id_col].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )
    df_labels["_key"] = df_labels["ID"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    )

    df_merged = pd.merge(df_features, df_labels, on="_key")
    df_merged.drop(columns=["_key"], inplace=True)

    print(f" Merge done - {len(df_merged):,} rows kept [{fmt_time(time.time()-t0)}]")
    print(f" Class distribution:\n{df_merged['label'].value_counts().to_string()}")
    return df_merged


# Drops non-numeric / identifier columns, encodes labels, and reports NaN counts (in short : some processing of reformating)
def prepare_features(df_merged):
    print("\n" + "=" * 60)
    print("STEP 2 - Preparing features")
    print("=" * 60)

    t0 = time.time()

    drop_cols = [c for c in df_merged.columns
                 if c.lower() in ("label", "id", "image_id", "filename", "file", "_key")]
    X     = df_merged.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y_raw = df_merged["label"]

    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print(f" NaN values in {len(nan_cols)} column(s) - will be imputed with median:")
        for col, count in nan_cols.items():
            print(f" {col:<35}: {count:,} NaNs ({100*count/len(X):.1f}%)")
    else:
        print("No NaN values found")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    print(f" Features: {X.shape[1]} columns, {X.shape[0]:,} samples")
    print(f" Classes : {list(encoder.classes_)}")
    print(f" Done [{fmt_time(time.time()-t0)}]")
    return X, y, encoder


# Stratified 80/20 split with balanced sample weights computed on the train set
def split_data(X, y):
    print("\n" + "=" * 60)
    print("STEP 3 - Train / test split  (80 / 20, stratified)")
    print("=" * 60)

    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    print(f" Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f" Done [{fmt_time(time.time()-t0)}]")
    return X_train, X_test, y_train, y_test, sample_weights


# Runs RandomizedSearchCV over an imputer -> scaler -> SVC pipeline, returns best estimator
def search_hyperparameters(X_train, y_train, sample_weights, n_iter=20):
    print("\n" + "=" * 60)
    print("STEP 4 - Hyperparameter search (RandomizedSearchCV)")
    print("=" * 60)
    print(f" {n_iter} combinations × 3 folds = up to {n_iter * 3} fits\n")

    # NaN imputation + feature scaling happen inside the pipeline to avoid data leakage
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(class_weight="balanced", cache_size=1000)),
    ])

    # svm__ prefix targets the SVC step inside the pipeline
    param_dist = {
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__kernel": ["rbf", "poly", "sigmoid"],
        "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "svm__degree": [2, 3, 4],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=3,
        refit=True,
        n_jobs=-1,
        random_state=42,
        verbose=3,
        error_score="raise",
    )

    t0 = time.time()
    search.fit(X_train, y_train, svm__sample_weight=sample_weights)

    print(f"\n Search done  [{fmt_time(time.time()-t0)}]")
    print(f" Best F1-macro (CV): {search.best_score_:.4f}")
    print(" Best parameters:")
    for k, v in search.best_params_.items():
        print(f"    {k:30s}: {v}")

    return search.best_estimator_


# Runs predictions, prints the classification report, and returns y_pred + macro F1
def evaluate(model, X_test, y_test, encoder):
    print("\n" + "=" * 60)
    print("STEP 5 - Evaluation on test set")
    print("=" * 60)

    t0     = time.time()
    y_pred = model.predict(X_test)
    print(f" Predictions done [{fmt_time(time.time()-t0)}]")

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n F1-Score Macro: {macro_f1:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(
        encoder.inverse_transform(y_test),
        encoder.inverse_transform(y_pred),
        zero_division=0,
    ))
    return y_pred, macro_f1


# Builds and saves a confusion matrix heatmap from predictions
def plot_confusion_matrix(y_test, y_pred, encoder, output_path):
    cm = confusion_matrix(y_test, y_pred)
    classes = encoder.classes_

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Purples)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(classes)), yticks=range(len(classes)),
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted label", ylabel="True label",
        title="SVM - Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f" Confusion matrix saved to {output_path}")


# Measures importance by the drop in F1-macro when each feature is randomly permuted
def compute_permutation_importance(model, X_test, y_test, feature_names):
    print("\n" + "=" * 60)
    print("STEP 6 - Permutation feature importance")
    print(f" {len(feature_names)} features to evaluate ...")
    print("=" * 60)

    baseline = f1_score(y_test, model.predict(X_test), average="macro")
    X_test_arr = np.array(X_test)
    importances = []
    t_total = time.time()

    print(f" Baseline F1-macro: {baseline:.4f}")

    for i, feat in enumerate(feature_names):
        t_feat = time.time()
        X_permuted = X_test_arr.copy()
        rng = np.random.default_rng(seed=i)
        X_permuted[:, i] = rng.permutation(X_permuted[:, i])

        score = f1_score(y_test, model.predict(X_permuted), average="macro")
        drop = baseline - score
        importances.append(drop)

        elapsed_total = time.time() - t_total
        eta = elapsed_total / (i + 1) * (len(feature_names) - i - 1)
        print(
            f"  [{i+1:>3}/{len(feature_names)}] {feat:<35} "
            f"drop={drop:+.4f}  "
            f"(this feat: {fmt_time(time.time()-t_feat)}, ETA: {fmt_time(eta)})"
        )

    return pd.Series(importances, index=feature_names).sort_values(ascending=True)


# Saves a horizontal bar chart of permutation importance scores
def plot_feature_importance(importances, output_path):
    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="purple")
    plt.title("SVM - Permutation Feature Importance")
    plt.xlabel("F1-macro drop when permuted")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f" Feature importance plot saved to {output_path}")


if __name__ == "__main__":
    script_start = time.time()
    print("=" * 60)
    print(" SVM CLASSIFIER — starting")
    print("=" * 60)

    if not os.path.isfile(FEATURES_CSV):
        raise FileNotFoundError(f"'{FEATURES_CSV}' not found. Run features.py first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load, merge, prepare
    df_features = load_features(FEATURES_CSV)
    df = merge_with_labels(df_features, METADATA_CSV)
    X, y, encoder = prepare_features(df)
    X_train, X_test, y_train, y_test, sample_weights = split_data(X, y)

    # Train
    best_model = search_hyperparameters(X_train, y_train, sample_weights, n_iter=20)

    # Evaluate
    y_pred, macro_f1 = evaluate(best_model, X_test, y_test, encoder)

    # Plots
    plot_confusion_matrix(
        y_test, y_pred, encoder,
        output_path=os.path.join(OUTPUT_DIR, "svm_confusion_matrix.png"),
    )
    importances = compute_permutation_importance(
        best_model, X_test, y_test,
        feature_names=list(X.columns),
    )
    plot_feature_importance(
        importances,
        output_path=os.path.join(OUTPUT_DIR, "svm_feature_importances.png"),
    )

    print("\n" + "=" * 60)
    print(f" ALL DONE - total runtime: {fmt_time(time.time()-script_start)}")
    print(f" Final F1-macro on test set: {macro_f1:.4f}")
    print("=" * 60)
