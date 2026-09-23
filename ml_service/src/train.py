import time
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src import config
from src.utils import sqli_tokenizer


# =============================
# Evaluation Helper
# =============================
def evaluate_model(name, pipeline, X_test, y_true):
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    roc  = roc_auc_score(y_true, y_proba)

    # False Positive Rate / False Negative Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print(f"\n📊 {name} Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"FPR      : {fpr:.4f}")
    print(f"FNR      : {fnr:.4f}")

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc,
        "FPR": fpr,
        "FNR": fnr,
    }


def main():
    # =============================
    # 1. Load Data
    # =============================
    if not config.DATA_PROCESSED.exists():
        print("❌ Data file not found. Run preprocess first.")
        return

    df = pd.read_csv(config.DATA_PROCESSED)

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    # =============================
    # 2. Dataset Info
    # =============================
    print("📊 Dataset Info:")
    print("Total samples:", len(df))
    print("\nLabel distribution:")
    print(y.value_counts())
    print("\nLabel ratio:")
    print(y.value_counts(normalize=True))

    # =============================
    # 3. Train/Test Split
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # =============================
    # 4. Define Pipelines
    # =============================

    # TF-IDF settings:
    #   - tokenizer=sqli_tokenizer : custom tokenizer for SQL syntax 
    #   - ngram_range=(1, 2)       : captures unigrams and bigrams (e.g. "union select")
    #   - min_df=5                 : ignore tokens appearing in fewer than 5 documents
    #   - max_features=2000        : limit feature space dimensionality

    # Random Forest Pipeline
    rf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=sqli_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=5,
            max_features=2000,
        )),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=config.RANDOM_STATE,
        )),
    ])

    # MLP Neural Network Pipeline
    mlp_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=sqli_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=5,
            max_features=2000,
        )),
        ("model", MLPClassifier(
            hidden_layer_sizes=(100, 100),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=config.RANDOM_STATE,
        )),
    ])

    # Naive Bayes Baseline 
    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=sqli_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=5,
            max_features=2000,
        )),
        ("model", MultinomialNB()),
    ])

    # =============================
    # 5. Train All Models
    # =============================
    print("\n🚀 Training Naive Bayes (baseline)...")
    start = time.time()
    nb_pipeline.fit(X_train, y_train)
    nb_train_time = time.time() - start

    print("🚀 Training Random Forest...")
    start = time.time()
    rf_pipeline.fit(X_train, y_train)
    rf_train_time = time.time() - start

    print("🚀 Training MLP Neural Network...")
    start = time.time()
    mlp_pipeline.fit(X_train, y_train)
    mlp_train_time = time.time() - start

    # =============================
    # 6. Evaluate All Models
    # =============================
    nb_metrics  = evaluate_model("Naive Bayes",   nb_pipeline,  X_test, y_test)
    rf_metrics  = evaluate_model("Random Forest", rf_pipeline,  X_test, y_test)
    mlp_metrics = evaluate_model("MLP",           mlp_pipeline, X_test, y_test)

    nb_metrics["Train Time (s)"]  = round(nb_train_time, 2)
    rf_metrics["Train Time (s)"]  = round(rf_train_time, 2)
    mlp_metrics["Train Time (s)"] = round(mlp_train_time, 2)

    # =============================
    # 7. Cross-Validation (5-fold)
    # =============================
    print("\n🔄 Running 5-fold Cross-Validation (this may take a while)...")

    nb_cv  = cross_val_score(nb_pipeline,  X, y, cv=config.CV_FOLDS, scoring="f1")
    rf_cv  = cross_val_score(rf_pipeline,  X, y, cv=config.CV_FOLDS, scoring="f1")
    mlp_cv = cross_val_score(mlp_pipeline, X, y, cv=config.CV_FOLDS, scoring="f1")

    print(f"Naive Bayes   CV F1: {nb_cv.mean():.4f} (+/- {nb_cv.std():.4f})")
    print(f"Random Forest CV F1: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")
    print(f"MLP           CV F1: {mlp_cv.mean():.4f} (+/- {mlp_cv.std():.4f})")

    nb_metrics["CV F1 Mean"]  = round(nb_cv.mean(), 4)
    nb_metrics["CV F1 Std"]   = round(nb_cv.std(), 4)
    rf_metrics["CV F1 Mean"]  = round(rf_cv.mean(), 4)
    rf_metrics["CV F1 Std"]   = round(rf_cv.std(), 4)
    mlp_metrics["CV F1 Mean"] = round(mlp_cv.mean(), 4)
    mlp_metrics["CV F1 Std"]  = round(mlp_cv.std(), 4)

    # =============================
    # 8. Save Comparison Table CSV
    # =============================
    config.OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame([nb_metrics, rf_metrics, mlp_metrics])
    comparison_df.to_csv(config.COMPARISON_TABLE_CSV, index=False)
    print(f"\n📋 Comparison table saved to {config.COMPARISON_TABLE_CSV}")

    # =============================
    # 9. Save All Models
    # =============================
    config.MODEL_PIPELINE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(nb_pipeline,  config.MODEL_NB)
    joblib.dump(rf_pipeline,  config.MODEL_RF)
    joblib.dump(mlp_pipeline, config.MODEL_MLP)
    print(f"💾 Saved nb_pipeline.pkl, rf_pipeline.pkl, mlp_pipeline.pkl to models/")

    # =============================
    # 10. Select and Save Best Model
    # =============================
    scores = {
        "Naive Bayes":   (nb_metrics["F1"],  nb_pipeline),
        "Random Forest": (rf_metrics["F1"],  rf_pipeline),
        "MLP":           (mlp_metrics["F1"], mlp_pipeline),
    }
    best_name = max(scores, key=lambda k: scores[k][0])
    best_model = scores[best_name][1]

    joblib.dump(best_model, config.MODEL_PIPELINE)
    print(f"\n🏆 Best Model: {best_name}")
    print(f"💾 Best model saved to {config.MODEL_PIPELINE}")


if __name__ == "__main__":
    main()