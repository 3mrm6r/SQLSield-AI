"""Path and configuration constants relative to the project root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================
# Data Paths
# =============================
DATA_RAW       = PROJECT_ROOT / "data" / "raw" / "data.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
DATA_SAMPLES   = PROJECT_ROOT / "data" / "samples" / "test_inputs.csv"

# =============================
# Model Paths
# =============================
# Best model (used in the site and predict.py)
MODEL_PIPELINE = PROJECT_ROOT / "models" / "pipeline.pkl"
MODEL_RF       = PROJECT_ROOT / "models" / "rf_pipeline.pkl"
MODEL_MLP      = PROJECT_ROOT / "models" / "mlp_pipeline.pkl"
MODEL_NB       = PROJECT_ROOT / "models" / "nb_pipeline.pkl"

# =============================
# Output Directories
# =============================
OUTPUTS_DIR            = PROJECT_ROOT / "outputs"
OUTPUT_REPORTS_DIR     = OUTPUTS_DIR / "reports"
OUTPUT_FIGURES_DIR     = OUTPUTS_DIR / "figures"
OUTPUT_PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# =============================
# Output Files
# =============================
METRICS_REPORT           = OUTPUT_REPORTS_DIR / "metrics.txt"
COMPARISON_TABLE_CSV     = OUTPUT_REPORTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_TXT   = OUTPUT_REPORTS_DIR / "feature_importance.txt"
OBFUSCATED_TEST_REPORT   = OUTPUT_REPORTS_DIR / "obfuscated_test_results.txt"

FIGURE_CONFUSION_MATRIX  = OUTPUT_FIGURES_DIR / "confusion_matrix.png"
FIGURE_DATA_DISTRIBUTION = OUTPUT_FIGURES_DIR / "data_distribution.png"
FIGURE_ROC_CURVE         = OUTPUT_FIGURES_DIR / "roc_curve.png"
FIGURE_FEATURE_IMPORTANCE = OUTPUT_FIGURES_DIR / "feature_importance.png"

SAMPLE_PREDICTIONS_CSV   = OUTPUT_PREDICTIONS_DIR / "sample_predictions.csv"

# =============================
# Constants
# =============================
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5