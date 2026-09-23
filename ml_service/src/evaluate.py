"""Evaluate trained pipeline and save metrics, figures, and predictions."""

import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src import config
from src.utils import ensure_dir, save_text_file, sqli_tokenizer


# Use Seaborn styling 
sns.set_theme(style="whitegrid")


# =============================
# Helper: Load Processed Data
# =============================
def load_processed():
    if not config.DATA_PROCESSED.exists():
        raise FileNotFoundError(
            f"Processed data not found at {config.DATA_PROCESSED}. "
            "Run preprocess.py first."
        )
    return pd.read_csv(config.DATA_PROCESSED)


# =============================
# Helper: Reproduce Same Split
# =============================
def same_split_as_train(df):
    X = df["text"].astype(str)
    y = df["label"].astype(int)
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


# =============================
# Main Evaluation Function
# =============================
def evaluate():
    # 1. Load model
    if not config.MODEL_PIPELINE.is_file():
        raise FileNotFoundError(
            f"No trained pipeline found at {config.MODEL_PIPELINE}. "
            "Run train.py first."
        )

    pipeline = joblib.load(config.MODEL_PIPELINE)

    # 2. Load data and reproduce the same test split used in training
    df = load_processed()
    _, X_test, _, y_test = same_split_as_train(df)

    # 3. Predict (and measure prediction time)
    start = time.time()
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start
    per_query_ms = (elapsed / len(X_test)) * 1000

    # =============================
    # 4. Compute Metrics
    # =============================
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_proba)

    # FPR/FNR
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    report_text = (
        "SQL Injection Detection - Evaluation Metrics\n"
        "============================================\n"
        f"Accuracy            : {acc:.4f}\n"
        f"Precision           : {prec:.4f}\n"
        f"Recall              : {rec:.4f}\n"
        f"F1-Score            : {f1:.4f}\n"
        f"ROC-AUC             : {roc:.4f}\n"
        f"False Positive Rate : {fpr:.4f}\n"
        f"False Negative Rate : {fnr:.4f}\n"
        f"\n"
        f"Confusion Matrix:\n"
        f"  True Negatives  (TN): {tn}\n"
        f"  False Positives (FP): {fp}\n"
        f"  False Negatives (FN): {fn}\n"
        f"  True Positives  (TP): {tp}\n"
        f"\n"
        f"Test samples            : {len(y_test)}\n"
        f"Total prediction time   : {elapsed:.3f}s\n"
        f"Avg time per query      : {per_query_ms:.3f}ms\n"
    )

    print(report_text)
    ensure_dir(config.OUTPUT_REPORTS_DIR)
    save_text_file(config.METRICS_REPORT, report_text)

    # =============================
    # 5. Confusion Matrix Figure
    # =============================
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "SQL Injection"],
        yticklabels=["Normal", "SQL Injection"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    ensure_dir(config.OUTPUT_FIGURES_DIR)
    fig.savefig(config.FIGURE_CONFUSION_MATRIX, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix to {config.FIGURE_CONFUSION_MATRIX}")

    # =============================
    # 6. ROC Curve Figure
    # =============================
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    ax_roc.plot(fpr_curve, tpr_curve, label=f"Model (AUC = {roc:.4f})", linewidth=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(config.FIGURE_ROC_CURVE, dpi=120)
    plt.close(fig_roc)
    print(f"Saved ROC curve to {config.FIGURE_ROC_CURVE}")

    # =============================
    # 7. Class Distribution Figure
    # =============================
    label_counts = df["label"].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=["Normal (0)", "SQL Injection (1)"],
        y=label_counts.values,
        palette=["#4C72B0", "#DD8452"],
        ax=ax2,
    )
    ax2.set_xlabel("Label")
    ax2.set_ylabel("Count")
    ax2.set_title("Class Distribution (Processed Data)")
    fig2.tight_layout()
    fig2.savefig(config.FIGURE_DATA_DISTRIBUTION, dpi=120)
    plt.close(fig2)
    print(f"Saved class distribution to {config.FIGURE_DATA_DISTRIBUTION}")

    # =============================
    # 8. Feature Importance (RF only)
    # =============================
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "token": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        top20 = fi_df.head(20)

        # Save text report
        fi_text = (
            "Top 20 Most Important Features (Random Forest)\n"
            "===============================================\n"
        )
        for _, row in top20.iterrows():
            fi_text += f"{row['token']:<30} {row['importance']:.6f}\n"
        save_text_file(config.FEATURE_IMPORTANCE_TXT, fi_text)
        print(f"Saved feature importance to {config.FEATURE_IMPORTANCE_TXT}")

        # Save bar chart figure
        fig_fi, ax_fi = plt.subplots(figsize=(9, 7))
        sns.barplot(
            x=top20["importance"][::-1],
            y=top20["token"][::-1],
            palette="viridis",
            ax=ax_fi,
        )
        ax_fi.set_title("Top 20 Most Important Features (Random Forest)")
        ax_fi.set_xlabel("Importance Score")
        ax_fi.set_ylabel("Token")
        fig_fi.tight_layout()
        fig_fi.savefig(config.FIGURE_FEATURE_IMPORTANCE, dpi=120)
        plt.close(fig_fi)
        print(f"Saved feature importance chart to {config.FIGURE_FEATURE_IMPORTANCE}")
    else:
        print("ℹ️  Best model is not a tree-based model — feature importance not available.")

    # =============================
    # 9. Save Sample Predictions CSV
    # =============================
    ensure_dir(config.OUTPUT_PREDICTIONS_DIR)
    predictions_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred,
        "confidence_sql_injection": y_proba,
    })
    predictions_df["text"] = predictions_df["text"].str[:500]
    predictions_df.to_csv(config.SAMPLE_PREDICTIONS_CSV, index=False)
    print(f"Saved sample predictions to {config.SAMPLE_PREDICTIONS_CSV}")


# =============================
# Entry Point
# =============================
if __name__ == "__main__":
    evaluate()