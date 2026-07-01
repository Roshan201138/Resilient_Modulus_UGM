from __future__ import annotations

import io
import os
import re
import sys
import types
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



APP_TITLE = "APP for resilient modulus prediction"
DEVELOPERS = "Developed by: Mohammad Jawed Roshan, António Gomes Correia, Ionut Dragos Moldovan, Miguel Azenha"

TARGET_NAME = "ResilientModulus_MPa"
TARGET_LABEL = "Resilient modulus"
TARGET_UNIT = "MPa"

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_DIR / "models"
MODEL_CACHE_DIR = APP_DIR / ".model_cache"

# When models are stored as GitHub Release assets, set this in Streamlit Secrets:
# GITHUB_RELEASE_BASE_URL = "https://github.com/<user>/<repo>/releases/download/v1.0"
DEFAULT_GITHUB_RELEASE_BASE_URL = (
    "https://github.com/Roshan201138/Resilient_Modulus_UGM/releases/download/v1.0"
)

MODEL_FILENAMES = [
    "KNN_search.joblib",
    "SVR_search.joblib",
    "Random_Forest_search.joblib",
    "DT_search.joblib",
    "LightGBM_search.joblib",
    "XGBoost_search.joblib",
    "ANN_search.joblib",
]

DEFAULT_FEATURES = [
    "Confining_pressure",
    "Loading_frequency",
    "Loading_cycle",
    "Deviator_stress",
    "Mean_stress",
    "Dissipated_energy",
    "Elastic_energy",
]

FEATURE_LABELS = {
    "Confining_pressure": "Confining pressure",
    "Loading_frequency": "Loading frequency",
    "Loading_cycle": "Loading cycle",
    "Deviator_stress": "Deviator stress",
    "Mean_stress": "Mean stress",
    "Dissipated_energy": "Dissipated energy",
    "Elastic_energy": "Elastic energy",
}

FEATURE_UNITS = {
    "Confining_pressure": "kPa",
    "Deviator_stress": "kPa",
    "Mean_stress": "kPa",
    "Loading_frequency": "Hz",
    "Loading_cycle": "cycles",
    "Dissipated_energy": "J/m³",
    "Elastic_energy": "J/m³",
}


TRAINING_RANGES = {
    "Confining_pressure": (20.0000, 50.0000),
    "Loading_frequency": (0.2000, 2.0000),
    "Loading_cycle": (1.0000, 480.0000),
    "Deviator_stress": (13.1611, 160.5767),
    "Mean_stress": (28.8315, 106.8510),
    "Dissipated_energy": (0.2249, 18.6458),
    "Elastic_energy": (0.7336, 67.5001),
    "ResilientModulus_MPa": (98.3242, 235.5393),
}

STANDARD_NAME_BY_FILE_KEY = {
    "knn": "KNN",
    "svr": "SVM",
    "svm": "SVM",
    "randomforest": "RF",
    "rf": "RF",
    "dt": "DT",
    "decisiontree": "DT",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "ann": "ANN",
}
MODEL_ORDER = ["KNN", "SVM", "RF", "DT", "LightGBM", "XGBoost", "ANN"]
EXCLUDED_MODEL_FILE_TOKENS = ("stack", "stacked", "hybrid")

PLOT_DPI = 300
ROUND_DECIMALS = 4


def format_training_range(feature: str) -> str:
    bounds = TRAINING_RANGES.get(feature)
    if not bounds:
        return ""
    return f"{bounds[0]:.4f}–{bounds[1]:.4f}"


def feature_input_label(feature: str) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    unit = FEATURE_UNITS.get(feature, "")
    label = f"{label} ({unit})" if unit else label
    range_text = format_training_range(feature)
    return f"{label} (training range: {range_text})" if range_text else label


def default_feature_value(feature: str) -> float:
    bounds = TRAINING_RANGES.get(feature)
    if not bounds:
        return 0.0
    return float((bounds[0] + bounds[1]) / 2.0)


# -----------------------------------------------------------------------------
# Model loading helpers
# -----------------------------------------------------------------------------

def friendly_load_error(filename: str, exc: Exception) -> str:
    msg = str(exc)
    lower = msg.lower()
    if "node array from the pickle has an incompatible dtype" in lower or "missing_go_to_left" in lower:
        return (
            "Incompatible scikit-learn version. This tree-based model was trained with scikit-learn 1.2.2. "
            "Use Python 3.10 and scikit-learn==1.2.2. "
            f"Details: {msg}"
        )
    if "euclideandistance" in lower or "_dist_metrics" in lower:
        return "Incompatible scikit-learn version for the KNN model. Use Python 3.10 and scikit-learn==1.2.2. " + msg
    if "could not locate class 'sequential'" in lower or "failed to restore serialized ann model" in lower or "keras" in lower or "protobuf" in lower:
        return (
            "Incompatible TensorFlow/Keras/protobuf environment for the ANN model. Use Python 3.10, tensorflow==2.10.0, "
            "protobuf==3.19.6, and set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python. "
            f"Details: {msg}"
        )
    return msg


def install_joblib_compatibility_modules() -> None:
    """Register lightweight classes required to unpickle GUI-trained joblib models."""
    if "models" not in sys.modules:
        models_mod = types.ModuleType("models")

        class KerasANNWrapper:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def _build_optimizer(self):
                import tensorflow as tf
                lr = float(getattr(self, "learning_rate_init", 0.001))
                name = str(getattr(self, "solver", "Adam")).strip().lower()
                constructors = {
                    "rmsprop": tf.keras.optimizers.RMSprop,
                    "adadelta": tf.keras.optimizers.Adadelta,
                    "adagrad": tf.keras.optimizers.Adagrad,
                    "adamax": tf.keras.optimizers.Adamax,
                    "nadam": tf.keras.optimizers.Nadam,
                    "ftrl": tf.keras.optimizers.Ftrl,
                    "adam": tf.keras.optimizers.Adam,
                    "sgd": tf.keras.optimizers.SGD,
                }
                if name == "adamw" and hasattr(tf.keras.optimizers, "AdamW"):
                    return tf.keras.optimizers.AdamW(learning_rate=lr)
                if name == "adafactor" and hasattr(tf.keras.optimizers, "Adafactor"):
                    return tf.keras.optimizers.Adafactor(learning_rate=lr)
                return constructors.get(name, tf.keras.optimizers.Adam)(learning_rate=lr)

            def __setstate__(self, state):
                self.__dict__.update(state)
                model_json = self.__dict__.pop("_serialized_model_json", None)
                model_weights = self.__dict__.pop("_serialized_model_weights", None)
                compile_cfg = self.__dict__.pop("_serialized_model_compile_config", None)
                if model_json is not None:
                    try:
                        import tensorflow as tf
                        model = tf.keras.models.model_from_json(model_json)
                        if model_weights is not None:
                            model.set_weights(model_weights)
                        try:
                            if compile_cfg:
                                model.compile_from_config(compile_cfg)
                            else:
                                model.compile(optimizer=self._build_optimizer(), loss="mse")
                        except Exception:
                            pass
                        self.model_ = model
                    except Exception as exc:
                        raise RuntimeError(
                            "Failed to restore serialized ANN model. Install TensorFlow compatible with the trained model. "
                            f"Details: {exc}"
                        ) from exc

            def __getstate__(self):
                return dict(self.__dict__)

            def predict(self, X):
                if not hasattr(self, "model_"):
                    raise RuntimeError("ANN model is not restored.")
                pred = self.model_.predict(np.asarray(X, dtype=np.float32), verbose=0)
                pred = np.asarray(pred)
                return pred.reshape(-1) if pred.ndim == 2 and pred.shape[1] == 1 else pred

            def predict_proba(self, X):
                if not hasattr(self, "model_"):
                    raise RuntimeError("ANN model is not restored.")
                pred = np.asarray(self.model_.predict(np.asarray(X, dtype=np.float32), verbose=0), dtype=float)
                if pred.ndim == 1 or pred.shape[1] == 1:
                    p1 = np.clip(pred.reshape(-1), 0.0, 1.0)
                    return np.column_stack([1.0 - p1, p1])
                row_sum = pred.sum(axis=1, keepdims=True)
                return np.divide(pred, row_sum, out=np.full_like(pred, 1.0 / pred.shape[1]), where=row_sum != 0)

        KerasANNWrapper.__module__ = "models"
        models_mod.KerasANNWrapper = KerasANNWrapper
        sys.modules["models"] = models_mod

    if "hybrid_models" not in sys.modules:
        # Hybrid models are excluded from this app, but these classes allow clean loading if a folder contains old files.
        hybrid_mod = types.ModuleType("hybrid_models")

        class HybridRuntimeBase:
            def __init__(self, *args, **kwargs):
                self.__dict__.update(kwargs)

        class HybridEnsembleRuntime(HybridRuntimeBase):
            pass

        class HybridResidualRuntime(HybridRuntimeBase):
            pass

        class HybridMultiBaseResidualRuntime(HybridRuntimeBase):
            pass

        class HybridOptimizerRuntime(HybridRuntimeBase):
            pass

        class HybridClassificationEnsembleRuntime(HybridRuntimeBase):
            pass

        class HybridClassificationStackingRuntime(HybridRuntimeBase):
            pass

        for cls in [HybridRuntimeBase, HybridEnsembleRuntime, HybridResidualRuntime, HybridMultiBaseResidualRuntime,
                    HybridOptimizerRuntime, HybridClassificationEnsembleRuntime, HybridClassificationStackingRuntime]:
            cls.__module__ = "hybrid_models"
            setattr(hybrid_mod, cls.__name__, cls)
        sys.modules["hybrid_models"] = hybrid_mod


install_joblib_compatibility_modules()


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def standard_model_name(path: Path, bundle: dict | None = None) -> str:
    candidates = [path.stem]
    if bundle:
        for key in ("model_name", "Model", "name"):
            if bundle.get(key):
                candidates.append(str(bundle[key]))
    key = normalize_text(" ".join(candidates))
    for token, standard in STANDARD_NAME_BY_FILE_KEY.items():
        if token in key:
            return standard
    return path.stem.replace("_search", "").replace("_", " ").strip()


def model_sort_key(name: str) -> tuple[int, str]:
    base = name.split(" (")[0]
    try:
        return MODEL_ORDER.index(base), name
    except ValueError:
        return len(MODEL_ORDER), name


def as_bundle(obj: Any, display_name: str, source_path: Path) -> dict:
    if isinstance(obj, dict):
        bundle = dict(obj)
    else:
        bundle = {
            "model": obj,
            "feature_cols": getattr(obj, "feature_cols", DEFAULT_FEATURES),
            "target_cols": getattr(obj, "target_cols", [TARGET_NAME]),
            "scaler": getattr(obj, "scaler", None),
            "scaler_name": getattr(obj, "scaler_name", "unknown"),
            "target_units": getattr(obj, "target_units", {TARGET_NAME: TARGET_UNIT}),
        }
    bundle["display_name"] = display_name
    bundle["source_path"] = str(source_path)
    bundle.setdefault("model_name", display_name)
    return bundle


def is_excluded_model_file(path: Path) -> bool:
    key = normalize_text(path.stem)
    return any(token in key for token in EXCLUDED_MODEL_FILE_TOKENS)


def discover_joblib_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists() or not model_dir.is_dir():
        return []
    return sorted(path for path in model_dir.glob("*.joblib") if not is_excluded_model_file(path))


def get_secret_or_env(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return str(os.environ.get(name, default)).strip()


def normalize_release_base_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def release_asset_url(base_url: str, filename: str) -> str:
    return f"{normalize_release_base_url(base_url)}/{quote(filename)}"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".download")
    if temp_path.exists():
        temp_path.unlink()
    urlretrieve(url, temp_path)
    temp_path.replace(destination)


def ensure_models_from_release(base_url: str, filenames: list[str] | None = None) -> tuple[Path, dict[str, str]]:
    """Download missing model files from a GitHub Release into a local cache folder."""
    filenames = filenames or MODEL_FILENAMES
    base_url = normalize_release_base_url(base_url)
    errors: dict[str, str] = {}

    if not base_url:
        return MODEL_CACHE_DIR, {
            "GitHub Release URL": (
                "Set GITHUB_RELEASE_BASE_URL in Streamlit Secrets, for example: "
                "https://github.com/<user>/<repo>/releases/download/v1.0"
            )
        }

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0.0, text="Checking trained model files...")
    total = max(1, len(filenames))
    for i, filename in enumerate(filenames, start=1):
        target = MODEL_CACHE_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            progress.progress(i / total, text=f"Model available: {filename}")
            continue
        try:
            progress.progress((i - 1) / total, text=f"Downloading {filename}...")
            download_file(release_asset_url(base_url, filename), target)
            progress.progress(i / total, text=f"Downloaded {filename}")
        except Exception as exc:
            errors[filename] = f"Could not download from {release_asset_url(base_url, filename)}. Details: {exc}"
    progress.empty()
    return MODEL_CACHE_DIR, errors


@st.cache_resource(show_spinner=False)
def load_models_from_folder(model_folder: str) -> tuple[dict[str, dict], dict[str, str]]:
    folder = Path(model_folder).expanduser().resolve()
    loaded: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for path in discover_joblib_files(folder):
        try:
            raw_obj = joblib.load(path)
            temp_bundle = raw_obj if isinstance(raw_obj, dict) else None
            display_name = standard_model_name(path, temp_bundle)
            final_name = display_name if display_name not in loaded else f"{display_name} ({path.stem})"
            loaded[final_name] = as_bundle(raw_obj, final_name, path)
        except Exception as exc:
            errors[path.name] = friendly_load_error(path.name, exc)
    loaded = dict(sorted(loaded.items(), key=lambda item: model_sort_key(item[0])))
    return loaded, errors


def get_feature_cols(bundle: dict) -> list[str]:
    model = bundle.get("model")
    return list(bundle.get("feature_cols") or getattr(model, "feature_cols", None) or DEFAULT_FEATURES)


def get_target_cols(bundle: dict) -> list[str]:
    model = bundle.get("model")
    return list(bundle.get("target_cols") or getattr(model, "target_cols", None) or [TARGET_NAME])


def read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or Excel file.")


def round_numeric_df(df: pd.DataFrame, decimals: int = ROUND_DECIMALS) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        round_numeric_df(df).to_excel(writer, index=False, sheet_name="Predictions")
    return buffer.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=PLOT_DPI, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    normalized = {normalize_text(c): c for c in df.columns}
    for name in possible_names:
        key = normalize_text(name)
        if key in normalized:
            return normalized[key]
    return None


def standardize_input_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    normalized_to_original = {normalize_text(c): c for c in df.columns}
    rename = {}
    missing = []
    for feature in feature_cols:
        candidates = [feature, FEATURE_LABELS.get(feature, feature), feature.replace("_", " "), feature.replace("_", "")]
        found = None
        for candidate in candidates:
            key = normalize_text(candidate)
            if key in normalized_to_original:
                found = normalized_to_original[key]
                break
        if found is None:
            missing.append(feature)
        else:
            rename[found] = feature
    if missing:
        raise ValueError("Missing required input column(s): " + ", ".join(missing))
    X = df.rename(columns=rename).loc[:, feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"These required columns contain missing or non-numeric values: {bad}")
    return X


def predict_one_model(bundle: dict, raw_input_df: pd.DataFrame) -> pd.DataFrame:
    model = bundle["model"]
    feature_cols = get_feature_cols(bundle)
    target_cols = get_target_cols(bundle)
    clean_raw_X = standardize_input_columns(raw_input_df, feature_cols)
    scaler = bundle.get("scaler")
    if scaler is not None:
        X_scaled = scaler.transform(clean_raw_X)
        model_input = pd.DataFrame(X_scaled, columns=feature_cols, index=clean_raw_X.index)
    else:
        model_input = clean_raw_X
    y_pred = model.predict(model_input)
    y = np.asarray(y_pred)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    pred_cols = [f"Predicted_{c}" for c in target_cols[: y.shape[1]]] or [f"Predicted_{TARGET_NAME}"]
    return pd.DataFrame(y, columns=pred_cols, index=raw_input_df.index)


def predict_multiple_models(bundles: dict[str, dict], raw_input_df: pd.DataFrame) -> pd.DataFrame:
    pred_frames = []
    for name, bundle in bundles.items():
        pred = predict_one_model(bundle, raw_input_df)
        first_col = pred.columns[0]
        pred_frames.append(pred[[first_col]].rename(columns={first_col: f"{name}_Predicted_{TARGET_NAME}"}))
    return pd.concat(pred_frames, axis=1)


# -----------------------------------------------------------------------------
# Metrics and plots
# -----------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "VAF": np.nan, "a10": np.nan}
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
    vaf = float((1 - np.var(err) / np.var(y_true)) * 100) if np.var(y_true) != 0 else np.nan
    safe_true = np.where(np.abs(y_true) < 1e-12, 1e-12, y_true)
    a10 = float(np.mean(np.abs(err / safe_true) <= 0.10) * 100)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MSE": mse, "VAF": vaf, "a10": a10}


def metrics_legend_text(metrics: dict[str, float]) -> str:
    return (
        f"R²={metrics['R2']:.4f}\n"
        f"RMSE={metrics['RMSE']:.4f} {TARGET_UNIT}\n"
        f"MAE={metrics['MAE']:.4f} {TARGET_UNIT}\n"
        f"MSE={metrics['MSE']:.4f}\n"
        f"VAF={metrics['VAF']:.4f}%\n"
        f"a10={metrics['a10']:.4f}%"
    )


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })


def create_measured_predicted_plot(y_true, y_pred, model_name: str):
    configure_plot_style()
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    abs_err = np.abs(y_pred - y_true)
    metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.2, 6.7))
    sc = ax.scatter(y_true, y_pred, c=abs_err, s=58, alpha=0.86, edgecolors="black", linewidths=0.35)
    low = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    high = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    if np.isclose(low, high):
        low -= 1.0
        high += 1.0
    pad = (high - low) * 0.08
    low -= pad
    high += pad
    ref = np.linspace(low, high, 200)
    ax.plot(ref, ref, "--", linewidth=1.5, label="1:1 line")
    ax.plot(ref, 1.2 * ref, ":", linewidth=1.2, label="+20% line")
    ax.plot(ref, 0.8 * ref, ":", linewidth=1.2, label="-20% line")
    ax.scatter([], [], label=metrics_legend_text(metrics), alpha=0)
    ax.set_xlabel(f"Measured {TARGET_LABEL} ({TARGET_UNIT})")
    ax.set_ylabel(f"Predicted {TARGET_LABEL} ({TARGET_UNIT})")
    ax.set_title(f"Measured-Predicted plot — {model_name}")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Absolute error ({TARGET_UNIT})")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    return fig, metrics


def create_error_plot(y_true, y_pred, model_name: str):
    configure_plot_style()
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    abs_err = np.abs(err)
    metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    idx = np.arange(1, len(err) + 1)
    sc = ax.scatter(idx, err, c=abs_err, s=52, alpha=0.86, edgecolors="black", linewidths=0.3)
    ax.axhline(0.0, linestyle="--", linewidth=1.4, label="Zero error")
    ax.scatter([], [], label=metrics_legend_text(metrics), alpha=0)
    ax.set_xlabel("Sample index")
    ax.set_ylabel(f"Prediction error ({TARGET_UNIT})")
    ax.set_title(f"Prediction error plot — {model_name}")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Absolute error ({TARGET_UNIT})")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig, metrics


def render_plots_for_predictions(results_df: pd.DataFrame, prediction_cols: list[str], measured_col: str | None, prefix: str) -> None:
    if measured_col is None:
        st.info("Measured target column was not found in the dataset, so measured-predicted and error plots are not displayed.")
        return
    y_true = pd.to_numeric(results_df[measured_col], errors="coerce")
    metrics_rows = []
    st.subheader("Dataset plots")
    for pred_col in prediction_cols:
        y_pred = pd.to_numeric(results_df[pred_col], errors="coerce")
        model_name = pred_col.replace(f"_Predicted_{TARGET_NAME}", "").replace(f"Predicted_{TARGET_NAME}", "Prediction")
        fig1, m = create_measured_predicted_plot(y_true, y_pred, model_name)
        fig2, _ = create_error_plot(y_true, y_pred, model_name)
        metrics_rows.append({"Model": model_name, **m})
        with st.expander(f"Plots — {model_name}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1, use_container_width=True)
                st.download_button(
                    f"Download measured-predicted plot ({model_name})",
                    fig_to_png_bytes(fig1),
                    file_name=f"{prefix}_{normalize_text(model_name)}_measured_predicted.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with col2:
                st.pyplot(fig2, use_container_width=True)
                st.download_button(
                    f"Download error plot ({model_name})",
                    fig_to_png_bytes(fig2),
                    file_name=f"{prefix}_{normalize_text(model_name)}_error_plot.png",
                    mime="image/png",
                    use_container_width=True,
                )
            plt.close(fig1)
            plt.close(fig2)
    if metrics_rows:
        st.subheader("Statistical metrics")
        st.dataframe(round_numeric_df(pd.DataFrame(metrics_rows)), use_container_width=True)


# -----------------------------------------------------------------------------
# Empirical models
# -----------------------------------------------------------------------------

MEASURED_TARGET_CANDIDATES = [
    TARGET_NAME,
    "Resilient modulus",
    "Resilient_Modulus",
    "Measured resilient modulus",
    "Measured_Resilient_Modulus",
    "Measured",
    "MR",
    "Mr",
    "M_R",
]

EMPIRICAL_MODEL_OPTIONS = [
    "MEPDG",
    "K-θ model",
    "Uzan model",
]


def stress_terms(confining_pressure, deviator_stress) -> tuple[np.ndarray, np.ndarray]:
    """Return triaxial stress invariants.

    sigma3 = confining pressure, q = deviator stress.
    theta = sigma1 + sigma2 + sigma3 = q + 3*sigma3.
    tau_oct = sqrt(2) * q / 3 for conventional triaxial compression.
    """
    sigma3 = np.asarray(confining_pressure, dtype=float)
    q = np.asarray(deviator_stress, dtype=float)
    theta = q + 3.0 * sigma3
    tau_oct = (np.sqrt(2.0) / 3.0) * np.abs(q)
    return theta, tau_oct


def measured_target_column(df: pd.DataFrame) -> str | None:
    return find_column(df, MEASURED_TARGET_CANDIDATES)


def empirical_required_features(models_to_run: list[str]) -> list[str]:
    if any(m in models_to_run for m in ["MEPDG", "K-θ model", "Uzan model"]):
        return ["Confining_pressure", "Deviator_stress"]
    return []


def _get_measured_y(df: pd.DataFrame) -> np.ndarray:
    y_col = measured_target_column(df)
    if y_col is None:
        raise ValueError(
            "Automatic calibration requires a measured resilient modulus column. "
            "Accepted names include ResilientModulus_MPa, Resilient modulus, MR, or Mr."
        )
    return pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)


def _standard_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = standardize_input_columns(df, ["Confining_pressure", "Deviator_stress"])
    y = _get_measured_y(df)
    return x, y


def _log_linear_fit(y: np.ndarray, columns: list[np.ndarray], min_rows: int, model_name: str) -> np.ndarray:
    arrays = [np.asarray(c, dtype=float) for c in columns]
    mask = np.isfinite(y) & (y > 0.0)
    for arr in arrays:
        mask &= np.isfinite(arr) & (arr > 0.0)
    if int(mask.sum()) < min_rows:
        raise ValueError(
            f"{model_name} calibration requires at least {min_rows} valid rows with positive stress terms "
            "and positive measured resilient modulus."
        )
    X = np.column_stack([np.ones(mask.sum())] + [np.log(arr[mask]) for arr in arrays])
    beta, *_ = np.linalg.lstsq(X, np.log(y[mask]), rcond=None)
    return beta


def confining_stress_prediction(confining_pressure, k1: float, k2: float) -> np.ndarray:
    sigma3 = np.asarray(confining_pressure, dtype=float)
    return float(k1) * np.power(np.maximum(sigma3, 1e-12), float(k2))


def k_theta_prediction(confining_pressure, deviator_stress, k1: float, k2: float) -> np.ndarray:
    theta, _ = stress_terms(confining_pressure, deviator_stress)
    return float(k1) * np.power(np.maximum(theta, 1e-12), float(k2))


def uzan_prediction(confining_pressure, deviator_stress, k1: float, k2: float, k3: float) -> np.ndarray:
    theta, _ = stress_terms(confining_pressure, deviator_stress)
    q = np.asarray(deviator_stress, dtype=float)
    return (
        float(k1)
        * np.power(np.maximum(theta, 1e-12), float(k2))
        * np.power(np.maximum(np.abs(q), 1e-12), float(k3))
    )


def mepdg_prediction(confining_pressure, deviator_stress, k1: float, k2: float, k3: float, pa: float) -> np.ndarray:
    theta, tau_oct = stress_terms(confining_pressure, deviator_stress)
    pa = max(float(pa), 1e-12)
    return (
        float(k1)
        * pa
        * np.power(np.maximum(theta / pa, 1e-12), float(k2))
        * np.power((tau_oct / pa) + 1.0, float(k3))
    )


def calibrate_confining_stress(df: pd.DataFrame) -> dict[str, float]:
    sigma3 = standardize_input_columns(df, ["Confining_pressure"])["Confining_pressure"].to_numpy(dtype=float)
    y = _get_measured_y(df)
    beta = _log_linear_fit(y, [sigma3], 2, "Confining stress model")
    return {"k1": float(np.exp(beta[0])), "k2": float(beta[1])}


def calibrate_k_theta(df: pd.DataFrame) -> dict[str, float]:
    x, y = _standard_xy(df)
    theta, _ = stress_terms(x["Confining_pressure"], x["Deviator_stress"])
    beta = _log_linear_fit(y, [theta], 2, "K-θ model")
    return {"k1": float(np.exp(beta[0])), "k2": float(beta[1])}


def calibrate_uzan(df: pd.DataFrame) -> dict[str, float]:
    x, y = _standard_xy(df)
    theta, _ = stress_terms(x["Confining_pressure"], x["Deviator_stress"])
    q = np.abs(x["Deviator_stress"].to_numpy(dtype=float))
    beta = _log_linear_fit(y, [theta, q], 3, "Uzan model")
    return {"k1": float(np.exp(beta[0])), "k2": float(beta[1]), "k3": float(beta[2])}


def calibrate_mepdg(df: pd.DataFrame, pa: float) -> dict[str, float]:
    x, y = _standard_xy(df)
    pa = max(float(pa), 1e-12)
    theta, tau_oct = stress_terms(x["Confining_pressure"], x["Deviator_stress"])
    beta = _log_linear_fit(y, [theta / pa, (tau_oct / pa) + 1.0], 3, "MEPDG")
    return {"k1": float(np.exp(beta[0]) / pa), "k2": float(beta[1]), "k3": float(beta[2]), "pa": pa}


def calibrate_empirical_models(df: pd.DataFrame, models_to_run: list[str], pa: float) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    params: dict[str, dict[str, float]] = {}
    rows = []
    if "MEPDG" in models_to_run:
        p = calibrate_mepdg(df, pa)
        params["MEPDG"] = p
        rows.append({"Model": "MEPDG", **p})
    if "K-θ model" in models_to_run:
        p = calibrate_k_theta(df)
        params["K-θ model"] = p
        rows.append({"Model": "K-θ model", **p})
    if "Uzan model" in models_to_run:
        p = calibrate_uzan(df)
        params["Uzan model"] = p
        rows.append({"Model": "Uzan model", **p})
    return params, pd.DataFrame(rows)


def empirical_predict(df: pd.DataFrame, models_to_run: list[str], params: dict[str, dict[str, float]]) -> pd.DataFrame:
    required = empirical_required_features(models_to_run)
    x = standardize_input_columns(df, required)
    out = pd.DataFrame(index=df.index)
    if "MEPDG" in models_to_run:
        if "MEPDG" not in params:
            raise ValueError("MEPDG coefficients are not calibrated yet.")
        p = params["MEPDG"]
        out[f"MEPDG_Predicted_{TARGET_NAME}"] = mepdg_prediction(
            x["Confining_pressure"], x["Deviator_stress"], p["k1"], p["k2"], p["k3"], p["pa"]
        )
    if "K-θ model" in models_to_run:
        if "K-θ model" not in params:
            raise ValueError("K-θ model coefficients are not calibrated yet.")
        p = params["K-θ model"]
        out[f"K-θ model_Predicted_{TARGET_NAME}"] = k_theta_prediction(
            x["Confining_pressure"], x["Deviator_stress"], p["k1"], p["k2"]
        )
    if "Uzan model" in models_to_run:
        if "Uzan model" not in params:
            raise ValueError("Uzan model coefficients are not calibrated yet.")
        p = params["Uzan model"]
        out[f"Uzan model_Predicted_{TARGET_NAME}"] = uzan_prediction(
            x["Confining_pressure"], x["Deviator_stress"], p["k1"], p["k2"], p["k3"]
        )
    return out


def render_empirical_calibration(models_to_run: list[str]) -> tuple[dict[str, dict[str, float]] | None, float]:
    st.subheader("Empirical model calibration")
    pa = st.number_input(
        "Reference pressure, Pa (same stress unit as confining/deviator stress)",
        value=101.325,
        format="%.4f",
        key="emp_pa",
    )
    calib_file = st.file_uploader(
        "Upload calibration dataset for empirical models (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="empirical_calibration_upload",
    )
    if calib_file is None:
        st.warning("Upload a calibration dataset to estimate the empirical model coefficients before single prediction.")
        return None, pa
    try:
        calib_df = read_table(calib_file)
        params, summary = calibrate_empirical_models(calib_df, models_to_run, pa)
        st.success("Empirical coefficients calibrated successfully.")
        st.dataframe(round_numeric_df(summary), use_container_width=True)

        pred_df = empirical_predict(calib_df, models_to_run, params)
        calib_output = pd.concat([calib_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        measured_col = measured_target_column(calib_output)
        render_plots_for_predictions(calib_output, list(pred_df.columns), measured_col, prefix="empirical_calibration")
        return params, pa
    except Exception as exc:
        st.error(str(exc))
        return None, pa


def render_logos_and_header() -> None:
    logo_dirs = [
        APP_DIR / "assets" / "logos",
        APP_DIR / "logos",
    ]

    logo_candidates = []
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".svg"}

    for folder in logo_dirs:
        if folder.exists():
            logo_candidates.extend(
                sorted(
                    [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_ext],
                    key=lambda p: p.name.lower()
                )
            )

    logo_candidates = logo_candidates[:4]

    if logo_candidates:
        cols = st.columns(len(logo_candidates))
        for col, logo in zip(cols, logo_candidates):
            with col:
                st.image(str(logo), width=230)
    else:
        st.warning("Logo files were not found. Please upload them to assets/logos/.")

    st.markdown(
        f"<h1 style='text-align:center; font-family:Times New Roman;'>{APP_TITLE}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center; font-family:Times New Roman; font-size:18px;'>{DEVELOPERS}</p>",
        unsafe_allow_html=True,
    )
    st.divider()


def render_model_loading_errors(errors: dict[str, str]) -> None:
    if not errors:
        return
    with st.expander("Model loading notes", expanded=False):
        st.write("Some joblib files in the model folder could not be loaded. Loaded models remain usable.")
        for filename, message in errors.items():
            st.warning(f"{filename}: {message}")


def render_required_columns(feature_cols: list[str]) -> None:
    with st.expander("Required input variables and training ranges", expanded=False):
        info = pd.DataFrame({
            "Required column name": feature_cols,
            "Display name": [FEATURE_LABELS.get(c, c) for c in feature_cols],
            "Training min": [TRAINING_RANGES.get(c, (np.nan, np.nan))[0] for c in feature_cols],
            "Training max": [TRAINING_RANGES.get(c, (np.nan, np.nan))[1] for c in feature_cols],
        })
        st.dataframe(round_numeric_df(info), use_container_width=True)


def render_single_ml_prediction(selected_bundles: dict[str, dict]) -> None:
    st.subheader("Single prediction")
    if not selected_bundles:
        st.warning("Select at least one ML model from the sidebar.")
        return
    first_bundle = next(iter(selected_bundles.values()))
    feature_cols = get_feature_cols(first_bundle)
    values = {}
    columns = st.columns(2)
    for i, feature in enumerate(feature_cols):
        with columns[i % 2]:
            values[feature] = st.number_input(feature_input_label(feature), value=default_feature_value(feature), format="%.4f", key=f"single_{feature}")
    if st.button("Predict resilient modulus", type="primary", use_container_width=True, key="ml_single_predict"):
        try:
            input_df = pd.DataFrame([values])
            pred_df = predict_multiple_models(selected_bundles, input_df)
            display = pred_df.T.reset_index()
            display.columns = ["Model", f"Predicted {TARGET_LABEL} ({TARGET_UNIT})"]
            display["Model"] = display["Model"].str.replace(f"_Predicted_{TARGET_NAME}", "", regex=False)
            st.dataframe(round_numeric_df(display), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


def render_batch_ml_prediction(selected_bundles: dict[str, dict]) -> None:
    st.subheader("Batch prediction")
    st.caption("The template uses midpoint values from the ML training ranges. Uploaded files may use values outside these ranges, but predictions are extrapolations.")
    if not selected_bundles:
        st.warning("Select at least one ML model from the sidebar.")
        return
    feature_cols = get_feature_cols(next(iter(selected_bundles.values())))
    template = pd.DataFrame([{feature: default_feature_value(feature) for feature in feature_cols}])
    template[TARGET_NAME] = 0.0
    st.download_button("Download CSV input template", round_numeric_df(template).to_csv(index=False).encode("utf-8"),
                       file_name="resilient_modulus_input_template.csv", mime="text/csv", use_container_width=True)
    uploaded = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx", "xls"], key="ml_batch_upload")
    if uploaded is None:
        return
    try:
        data = read_table(uploaded)
        st.write("Dataset preview")
        st.dataframe(round_numeric_df(data.head(20)), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))
        return
    if st.button("Run batch prediction", type="primary", use_container_width=True, key="ml_batch_predict"):
        try:
            pred_df = predict_multiple_models(selected_bundles, data)
            output = pd.concat([data.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
            st.success(f"Predicted {len(output)} row(s).")
            st.dataframe(round_numeric_df(output), use_container_width=True)
            st.download_button("Download predictions as Excel", excel_bytes(output),
                               file_name="resilient_modulus_ml_predictions.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            measured_col = measured_target_column(output)
            render_plots_for_predictions(output, list(pred_df.columns), measured_col, prefix="ml")
        except Exception as exc:
            st.error(str(exc))


def render_single_empirical_prediction(models_to_run: list[str], params: dict[str, dict[str, float]] | None) -> None:
    st.subheader("Single empirical prediction")
    if not models_to_run:
        st.warning("Select at least one empirical model from the sidebar.")
        return
    if params is None:
        st.warning("Upload a calibration dataset first. Single empirical prediction needs calibrated coefficients.")
        return
    required = empirical_required_features(models_to_run)
    values = {}
    columns = st.columns(max(1, min(2, len(required))))
    for i, feature in enumerate(required):
        with columns[i % len(columns)]:
            values[feature] = st.number_input(feature_input_label(feature), value=default_feature_value(feature), format="%.4f", key=f"emp_single_{feature}")
    if st.button("Predict resilient modulus", type="primary", use_container_width=True, key="emp_single_predict"):
        data = pd.DataFrame([values])
        pred = empirical_predict(data, models_to_run, params)
        display = pred.T.reset_index()
        display.columns = ["Model", f"Predicted {TARGET_LABEL} ({TARGET_UNIT})"]
        display["Model"] = display["Model"].str.replace(f"_Predicted_{TARGET_NAME}", "", regex=False)
        st.dataframe(round_numeric_df(display), use_container_width=True)


def render_batch_empirical_prediction(models_to_run: list[str], params: dict[str, dict[str, float]] | None, pa: float) -> None:
    st.subheader("Batch prediction")
    st.caption("The template uses midpoint values from the available ML training ranges as guidance.")
    if not models_to_run:
        st.warning("Select at least one empirical model from the sidebar.")
        return
    template_cols = empirical_required_features(models_to_run)
    template = pd.DataFrame([{col: default_feature_value(col) for col in template_cols}])
    template[TARGET_NAME] = 0.0
    st.download_button("Download CSV input template", round_numeric_df(template).to_csv(index=False).encode("utf-8"),
                       file_name="resilient_modulus_empirical_template.csv", mime="text/csv", use_container_width=True)
    uploaded = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx", "xls"], key="emp_batch_upload")
    if uploaded is None:
        return
    try:
        data = read_table(uploaded)
        st.write("Dataset preview")
        st.dataframe(round_numeric_df(data.head(20)), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))
        return
    if st.button("Run empirical batch prediction", type="primary", use_container_width=True, key="emp_batch_predict"):
        try:
            run_params = params
            if run_params is None:
                if measured_target_column(data) is None:
                    raise ValueError("Upload a calibration dataset or include measured resilient modulus in the batch file for automatic calibration.")
                run_params, summary = calibrate_empirical_models(data, models_to_run, pa)
                st.success("Empirical coefficients calibrated from the uploaded batch dataset.")
                st.dataframe(round_numeric_df(summary), use_container_width=True)
            pred_df = empirical_predict(data, models_to_run, run_params)
            output = pd.concat([data.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
            st.success(f"Predicted {len(output)} row(s).")
            st.dataframe(round_numeric_df(output), use_container_width=True)
            st.download_button("Download predictions as Excel", excel_bytes(output),
                               file_name="resilient_modulus_empirical_predictions.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            measured_col = measured_target_column(output)
            render_plots_for_predictions(output, list(pred_df.columns), measured_col, prefix="empirical")
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: 'Times New Roman', Times, serif; }
        .stButton button { font-family: 'Times New Roman', Times, serif; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_logos_and_header()
    page = st.sidebar.radio("Application mode", ["Trained ML models", "Empirical models"], index=0)

    if page == "Trained ML models":
        # Model files are downloaded automatically from the configured GitHub Release.
        # The URL is intentionally not shown in the user interface.
        release_url = get_secret_or_env("GITHUB_RELEASE_BASE_URL", DEFAULT_GITHUB_RELEASE_BASE_URL)

        model_folder, download_errors = ensure_models_from_release(release_url)
        loaded_models, load_errors = load_models_from_folder(str(model_folder))
        errors = {**download_errors, **load_errors}
        render_model_loading_errors(errors)

        if not loaded_models:
            st.error("No trained ML model could be loaded. Check the GitHub Release URL and model asset filenames.")
            return

        st.sidebar.header("ML models")
        selected_names = st.sidebar.multiselect("Choose one or more models", list(loaded_models.keys()), default=list(loaded_models.keys())[:1])
        selected_bundles = {name: loaded_models[name] for name in selected_names}
        tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])
        with tab_single:
            render_single_ml_prediction(selected_bundles)
        with tab_batch:
            render_batch_ml_prediction(selected_bundles)

    else:
        st.sidebar.header("Empirical models")
        empirical_models = st.sidebar.multiselect(
            "Choose one or more empirical models",
            EMPIRICAL_MODEL_OPTIONS,
            default=EMPIRICAL_MODEL_OPTIONS,
        )
        params, pa = render_empirical_calibration(empirical_models)
        render_single_empirical_prediction(empirical_models, params)


if __name__ == "__main__":
    main()
