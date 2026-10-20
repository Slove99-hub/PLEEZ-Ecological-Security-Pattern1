
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "samples" / "samples.csv"
OUTPUT_DIR = BASE_DIR / "results"

FEATURE_COLS = [
    "DEM",
    "Slope",
    "RoadDist",
    "NDVI",
    "PopDensity",
    "Nightlight",
    "WaterDist"
]

LABEL_COL = "Label"
RANDOM_SEED = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 1800,
    "max_depth": 5,
    "learning_rate": 0.015,
    "reg_lambda": 7,
    "reg_alpha": 4.5,
    "gamma": 1.9,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1
}


def load_data():
    df = pd.read_csv(DATA_PATH)

    required_cols = FEATURE_COLS + [LABEL_COL]

    if not set(required_cols).issubset(df.columns):
        raise ValueError("Required columns are missing.")

    df = df[required_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if df[LABEL_COL].isna().any():
        raise ValueError("Missing class labels detected.")

    if not df[LABEL_COL].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    df[FEATURE_COLS] = df[FEATURE_COLS].apply(
        pd.to_numeric, errors="coerce"
    )

    if df[FEATURE_COLS].isna().any().any():
        raise ValueError("Missing feature values detected.")

    df[LABEL_COL] = df[LABEL_COL].astype(int)

    return df


def create_model(scale_pos_weight=1.0):
    return XGBClassifier(
        **MODEL_PARAMS,
        scale_pos_weight=scale_pos_weight
    )


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y, probabilities)
    }

    return metrics, predictions, probabilities


def plot_roc(y_train, train_proba, y_test, test_proba):
    fpr_train, tpr_train, _ = roc_curve(
        y_train, train_proba
    )
    fpr_test, tpr_test, _ = roc_curve(
        y_test, test_proba
    )

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr_train, tpr_train,
        label=f"Training (AUC = {train_auc:.4f})"
    )
    plt.plot(
        fpr_test, tpr_test,
        label=f"Testing (AUC = {test_auc:.4f})"
    )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_feature_importance(model):
    importance = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(8, 6))
    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low", "High"])
    ax.set_yticklabels(["Low", "High"])

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLS,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLS,
        index=X_test.index
    )

    class_counts = y_train.value_counts()
    scale_pos_weight = (
        class_counts[0] / class_counts[1]
    )

    model = create_model(scale_pos_weight)
    model.fit(X_train_scaled, y_train)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_cols": FEATURE_COLS
        },
        OUTPUT_DIR / "xgb_model.pkl"
    )

    joblib.dump(
        scaler,
        OUTPUT_DIR / "scaler.pkl"
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    # Each fold fits its own scaler.
    from sklearn.pipeline import Pipeline

    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", create_model(scale_pos_weight))
    ])

    cv_scores = cross_val_score(
        cv_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )

    pd.DataFrame({
        "Fold": np.arange(1, CV_FOLDS + 1),
        "ROC_AUC": cv_scores
    }).to_csv(
        OUTPUT_DIR / "cross_validation.csv",
        index=False
    )

    train_metrics, train_pred, train_proba = (
        evaluate_model(
            model, X_train_scaled, y_train
        )
    )

    test_metrics, test_pred, test_proba = (
        evaluate_model(
            model, X_test_scaled, y_test
        )
    )

    metrics_df = pd.DataFrame({
        "Training": train_metrics,
        "Testing": test_metrics
    }).T

    metrics_df.to_csv(
        OUTPUT_DIR / "model_metrics.csv"
    )

    with open(
        OUTPUT_DIR / "classification_reports.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("Training Classification Report\n")
        file.write(
            classification_report(
                y_train, train_pred, digits=4
            )
        )

        file.write("\nTesting Classification Report\n")
        file.write(
            classification_report(
                y_test, test_pred, digits=4
            )
        )

    plot_roc(
        y_train, train_proba,
        y_test, test_proba
    )

    plot_feature_importance(model)

    plot_confusion_matrix(
        y_test, test_pred
    )

    print(metrics_df.round(4))
    print(f"Cross-validation AUC: {cv_scores.mean():.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
