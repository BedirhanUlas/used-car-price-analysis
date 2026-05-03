"""
Used Car Price Analysis — Regression Pipeline

Predicts listing prices for used vehicles on Craigslist using
Linear Regression, Ridge, and Lasso regularization.
Dataset: Kaggle Craigslist Used Cars (~426K records).
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "vehicles.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.2

PRICE_MIN = 500
PRICE_MAX = 150_000
ODOMETER_MAX = 500_000

NUMERIC_FEATURES = ["year", "odometer"]
CATEGORICAL_FEATURES = [
    "manufacturer", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "type", "paint_color", "state",
]


def load_data(path: Path) -> pd.DataFrame:
    """Load Craigslist vehicles dataset.

    Args:
        path: Path to vehicles.csv.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If CSV is not found at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download from: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data"
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def explore(df: pd.DataFrame) -> None:
    """Log key dataset statistics."""
    logger.info(f"Price range: ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")
    logger.info(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    logger.info(f"Top manufacturers:\n{df['manufacturer'].value_counts().head(10)}")


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Clean, filter, and encode the vehicles dataset.

    Strategy:
        - Drop duplicate listings and columns with >50% missing values
        - Filter price outliers (PRICE_MIN – PRICE_MAX) and odometer outliers
        - Impute missing categoricals with mode, numerics with median
        - One-hot encode categorical features
        - StandardScaler on numeric features

    Returns:
        X (features array), y (price array), feature_names (list).
    """
    df = df.copy()

    target_col = "price"
    y_raw = df[target_col]

    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target_col]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Filter nonsensical prices and odometer readings
    price_mask = (df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)
    if "odometer" in df.columns:
        odometer_mask = df["odometer"].fillna(0) <= ODOMETER_MAX
        df = df[price_mask & odometer_mask]
    else:
        df = df[price_mask]

    df = df.drop_duplicates()
    logger.info(f"After filtering: {len(df):,} rows")

    # Fill missing values
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    y = df[target_col].values
    df = df.drop(columns=[target_col])

    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=available_cat, drop_first=True)
    feature_names = df_encoded.columns.tolist()

    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    scaler = StandardScaler()
    df_encoded[available_num] = scaler.fit_transform(df_encoded[available_num])

    X = df_encoded.values.astype(float)
    logger.info(f"Features after encoding: {X.shape[1]}")
    return X, y, feature_names


def build_models() -> Dict[str, object]:
    """Return regression models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge (α=1.0)": Ridge(alpha=1.0),
        "Lasso (α=1.0)": Lasso(alpha=1.0, max_iter=5000),
    }


def evaluate_model(
    name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Fit a model and return regression evaluation metrics.

    Args:
        name: Human-readable model label.
        model: Unfitted sklearn estimator.
        X_train, X_test, y_train, y_test: Split arrays.

    Returns:
        Dict with MAE, RMSE, R², train_time_s.
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = round(time.perf_counter() - t0, 2)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "Model": name,
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "R²": round(r2, 4),
        "Train Time (s)": train_time,
    }


def plot_feature_importance(model, feature_names: List[str], save_dir: Path, top_n: int = 20) -> None:
    """Save a bar chart of top feature importances (coefficients) for linear models."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "coef_"):
        return

    importances = pd.Series(np.abs(model.coef_), index=feature_names)
    top = importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    top.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("muted")[0], edgecolor="black")
    ax.set_title(f"Top {top_n} Feature Importances (|coefficient|)", fontsize=12)
    ax.set_xlabel("|Coefficient|")
    plt.tight_layout()
    out = save_dir / "feature_importance.png"
    fig.savefig(out, dpi=150)
    logger.info(f"Feature importance chart saved → {out}")
    plt.close(fig)


def plot_results(results_df: pd.DataFrame, save_dir: Path) -> None:
    """Save a grouped bar chart comparing MAE and RMSE across models."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["MAE ($)", "RMSE ($)"]):
        bars = ax.bar(
            results_df["Model"],
            results_df[metric],
            color=sns.color_palette("muted"),
            edgecolor="black",
        )
        ax.bar_label(bars, fmt="$%.0f", padding=3)
        ax.set_title(f"{metric} by Model")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.suptitle("Used Car Price Prediction — Model Comparison", fontsize=13)
    plt.tight_layout()
    out = save_dir / "model_comparison.png"
    fig.savefig(out, dpi=150)
    logger.info(f"Comparison chart saved → {out}")
    plt.close(fig)


def main() -> None:
    logger.info("=== Used Car Price Regression Pipeline ===")

    df = load_data(DATA_PATH)
    explore(df)

    X, y, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    models = build_models()
    results = []
    best_model, best_mae = None, float("inf")

    for name, model in models.items():
        logger.info(f"Training: {name}...")
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        logger.info(f"  MAE: ${metrics['MAE ($)']:,.0f} | RMSE: ${metrics['RMSE ($)']:,.0f} | R²: {metrics['R²']}")
        if metrics["MAE ($)"] < best_mae:
            best_mae = metrics["MAE ($)"]
            best_model = model

    results_df = pd.DataFrame(results)
    logger.info("\n=== Results ===\n" + results_df.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    plot_results(results_df, RESULTS_DIR)
    plot_feature_importance(best_model, feature_names, RESULTS_DIR)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
