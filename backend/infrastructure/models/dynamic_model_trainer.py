"""Dynamic AML Model Trainer for Custom Labeled Datasets (Mode B).

Trains, evaluates, and registers custom models on user-provided labeled transaction CSVs
without modifying existing authoritative federated IBM AML checkpoints.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from infrastructure.data.schemas.universal_schema_detector import (
    UniversalSchemaDetector,
    UniversalSchemaReport,
)
from infrastructure.models.registry.universal_model_registry import (
    ModelSchemaMetadata,
    UniversalModelRegistry,
)

logger = logging.getLogger(__name__)


class DynamicModelPipeline:
    """Encapsulates fitted feature preprocessors and trained tabular classifier."""

    def __init__(
        self,
        feature_columns: List[str],
        numerical_columns: List[str],
        categorical_columns: List[str],
        num_imputer: SimpleImputer,
        num_scaler: StandardScaler,
        cat_imputer: SimpleImputer,
        cat_encoder: OrdinalEncoder,
        model: Any,
        decision_threshold: float,
    ) -> None:
        self.feature_columns = feature_columns
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.num_imputer = num_imputer
        self.num_scaler = num_scaler
        self.cat_imputer = cat_imputer
        self.cat_encoder = cat_encoder
        self.model = model
        self.decision_threshold = decision_threshold

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw input DataFrame into scaled model-ready feature matrix."""
        n_samples = len(df)
        if n_samples == 0:
            return np.empty((0, len(self.feature_columns)), dtype=np.float32)

        parts: List[np.ndarray] = []

        # Process numerical columns
        if self.numerical_columns:
            # Ensure columns exist, fill missing with NaN
            num_sub = pd.DataFrame(index=df.index)
            for c in self.numerical_columns:
                num_sub[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

            imputed_num = self.num_imputer.transform(num_sub)
            scaled_num = self.num_scaler.transform(imputed_num)
            parts.append(scaled_num.astype(np.float32))

        # Process categorical columns
        if self.categorical_columns:
            cat_sub = pd.DataFrame(index=df.index)
            for c in self.categorical_columns:
                cat_sub[c] = df[c].astype(str) if c in df.columns else "missing"

            imputed_cat = self.cat_imputer.transform(cat_sub)
            encoded_cat = self.cat_encoder.transform(imputed_cat)
            parts.append(encoded_cat.astype(np.float32))

        if not parts:
            raise ValueError("No features available to transform.")

        return np.hstack(parts)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict positive laundering probabilities for input DataFrame."""
        X = self.transform(df)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.shape[1] == 2:
                return probs[:, 1]
            return probs[:, 0]
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            # Sigmoid mapping
            return 1.0 / (1.0 + np.exp(-scores))
        else:
            return self.model.predict(X).astype(float)


class DynamicModelTrainer:
    """Trains custom AML classification models from labeled CSV files."""

    def __init__(self) -> None:
        self.schema_detector = UniversalSchemaDetector()
        self.model_registry = UniversalModelRegistry()

    def train_custom_model(
        self,
        df: pd.DataFrame,
        model_name: str,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        model_type: str = "HistGradientBoosting",  # "HistGradientBoosting" or "RandomForest"
    ) -> Tuple[ModelSchemaMetadata, Dict[str, Any]]:
        """Train a new AML classifier on a labeled dataset and register it."""
        t0 = time.perf_counter()

        if len(df) < 5:
            raise ValueError(f"Dataset contains only {len(df)} row(s). Minimum 5 rows required for training.")

        # 1. Infer schema
        report: UniversalSchemaReport = self.schema_detector.inspect_dataframe(df)

        target_col = target_column or report.target_column_name
        if not target_col or target_col not in df.columns:
            raise ValueError(
                "No target column identified for supervised training. "
                "Please specify a target column containing binary laundering/fraud labels (0/1)."
            )

        # 2. Validate target labels
        raw_target = pd.to_numeric(df[target_col], errors="coerce")
        valid_mask = raw_target.notna() & raw_target.isin([0, 1])
        if valid_mask.sum() < len(df) * 0.9:
            raise ValueError(
                f"Target column '{target_col}' must contain binary indicators (0 for legitimate, 1 for laundering/fraud)."
            )

        clean_df = df[valid_mask].copy()
        y = raw_target[valid_mask].astype(int).to_numpy()

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Training dataset requires both positive and negative examples. Found only class {unique_classes}."
            )

        # 3. Determine feature columns (exclude target and high-cardinality IDs)
        if feature_columns:
            chosen_features = [c for c in feature_columns if c in clean_df.columns and c != target_col]
        else:
            excluded = set(report.id_columns) | {target_col}
            chosen_features = [c for c in clean_df.columns if c not in excluded]

        if not chosen_features:
            raise ValueError("No feature columns remaining after excluding target and identifier columns.")

        num_cols = [c for c in chosen_features if c in report.numerical_columns or pd.api.types.is_numeric_dtype(clean_df[c])]
        cat_cols = [c for c in chosen_features if c not in num_cols]

        # 4. Train / Validation Split (Stratified 80 / 20)
        train_idx, val_idx = train_test_split(
            np.arange(len(clean_df)),
            test_size=0.2,
            random_state=42,
            stratify=y if min((y == 0).sum(), (y == 1).sum()) >= 2 else None,
        )

        df_train = clean_df.iloc[train_idx]
        y_train = y[train_idx]
        df_val = clean_df.iloc[val_idx]
        y_val = y[val_idx]

        # 5. Fit Preprocessors on Training Split Only (Zero Leakage)
        num_imputer = SimpleImputer(strategy="median")
        num_scaler = StandardScaler()
        cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        train_parts: List[np.ndarray] = []
        val_parts: List[np.ndarray] = []

        if num_cols:
            train_num = df_train[num_cols].apply(pd.to_numeric, errors="coerce")
            val_num = df_val[num_cols].apply(pd.to_numeric, errors="coerce")

            X_num_train = num_scaler.fit_transform(num_imputer.fit_transform(train_num))
            X_num_val = num_scaler.transform(num_imputer.transform(val_num))

            train_parts.append(X_num_train.astype(np.float32))
            val_parts.append(X_num_val.astype(np.float32))

        if cat_cols:
            train_cat = df_train[cat_cols].astype(str)
            val_cat = df_val[cat_cols].astype(str)

            X_cat_train = cat_encoder.fit_transform(cat_imputer.fit_transform(train_cat))
            X_cat_val = cat_encoder.transform(cat_imputer.transform(val_cat))

            train_parts.append(X_cat_train.astype(np.float32))
            val_parts.append(X_cat_val.astype(np.float32))

        X_train = np.hstack(train_parts)
        X_val = np.hstack(val_parts)

        # 6. Fit Model with Class Imbalance Weighting
        pos_weight = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))

        if model_type.lower() == "randomforest":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                class_weight="balanced",
                random_state=42,
            )

        clf.fit(X_train, y_train)

        # 7. Evaluate on Validation Split and Tune Decision Threshold
        if hasattr(clf, "predict_proba"):
            probs_val = clf.predict_proba(X_val)[:, 1]
        else:
            probs_val = clf.decision_function(X_val)
            probs_val = 1.0 / (1.0 + np.exp(-probs_val))

        # Threshold optimization on validation split (maximize F1)
        candidate_thresholds = np.linspace(0.1, 0.9, 17)
        best_th = 0.5
        best_f1 = -1.0

        for th in candidate_thresholds:
            preds_th = (probs_val >= th).astype(int)
            f1_th = f1_score(y_val, preds_th, zero_division=0)
            if f1_th > best_f1:
                best_f1 = f1_th
                best_th = float(th)

        val_preds = (probs_val >= best_th).astype(int)

        # Validation metrics
        try:
            val_roc_auc = float(roc_auc_score(y_val, probs_val))
        except Exception:
            val_roc_auc = 0.5

        try:
            val_pr_auc = float(average_precision_score(y_val, probs_val))
        except Exception:
            val_pr_auc = float(np.mean(y_val))

        val_f1 = float(f1_score(y_val, val_preds, zero_division=0))
        val_prec = float(precision_score(y_val, val_preds, zero_division=0))
        val_rec = float(recall_score(y_val, val_preds, zero_division=0))
        cm = confusion_matrix(y_val, val_preds).tolist()

        training_metrics = {
            "validation_roc_auc": round(val_roc_auc, 4),
            "validation_pr_auc": round(val_pr_auc, 4),
            "validation_f1": round(val_f1, 4),
            "validation_precision": round(val_prec, 4),
            "validation_recall": round(val_rec, 4),
            "confusion_matrix": cm,
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "class_imbalance_ratio": round(pos_weight, 2),
            "training_time_seconds": round(time.perf_counter() - t0, 3),
        }

        # 8. Package Pipeline and Register Model
        pipeline = DynamicModelPipeline(
            feature_columns=chosen_features,
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
            num_imputer=num_imputer,
            num_scaler=num_scaler,
            cat_imputer=cat_imputer,
            cat_encoder=cat_encoder,
            model=clf,
            decision_threshold=best_th,
        )

        model_id = f"custom_{re.sub(r'[^a-zA-Z0-9_]', '_', model_name.lower())}_{int(time.time())}"

        metadata = ModelSchemaMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            dataset_name=f"Custom ({model_name})",
            target_column=target_col,
            feature_columns=chosen_features,
            categorical_columns=cat_cols,
            numerical_columns=num_cols,
            preprocessing_version="dynamic_v1",
            model_version="1.0.0",
            input_shape=[len(chosen_features)],
            class_labels=["Legitimate", "Laundering"],
            training_timestamp=datetime.now(timezone.utc).isoformat(),
            frozen_threshold=best_th,
            is_sequence_model=False,
            required_sequence_fields=[],
            description=f"User-trained AML model '{model_name}' trained on {len(df)} rows across {len(chosen_features)} features.",
            is_builtin=False,
            training_metrics=training_metrics,
        )

        self.model_registry.register_dynamic_model(
            metadata=metadata,
            model_artifact=pipeline,
            preprocessor_artifact=pipeline,
        )

        return metadata, training_metrics

