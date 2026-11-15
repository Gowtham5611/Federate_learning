"""Universal Model Schema Registry for FedSecureAI.

Maintains metadata and preprocessing pipelines for:
1. Pre-trained Authoritative Federated Models (HI-Small LSTM/GRU/BiLSTM, LI-Medium).
2. Dynamic User-Trained AML Models created from custom uploaded labeled datasets.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[3]
_ROOT_DIR = _BACKEND_DIR.parent
BASE_DIR = _BACKEND_DIR if (_BACKEND_DIR / "results").exists() else _ROOT_DIR
DYNAMIC_MODELS_DIR = BASE_DIR / "results/dynamic_models"


@dataclass
class ModelSchemaMetadata:
    """Standardized metadata describing a model, its schema, and its expected inputs."""

    model_id: str
    model_name: str
    model_type: str  # "LSTM", "GRU", "BiLSTM", "Ensemble", "RandomForest", "MLP", "LogisticRegression"
    dataset_name: str  # "HI-Small", "LI-Medium", or custom dataset name
    target_column: str
    feature_columns: List[str]
    categorical_columns: List[str]
    numerical_columns: List[str]
    preprocessing_version: str
    model_version: str
    input_shape: List[int]  # e.g., [10, 57] for recurrent, [57] for tabular
    class_labels: List[str]  # e.g., ["Legitimate", "Laundering"]
    training_timestamp: str
    frozen_threshold: float
    is_sequence_model: bool
    required_sequence_fields: List[str] = field(default_factory=list)
    description: str = ""
    is_builtin: bool = True
    checkpoint_path: Optional[str] = None
    expected_sha256: Optional[str] = None
    pipeline_artifact_path: Optional[str] = None
    training_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to serializable dict."""
        res = asdict(self)
        res["decision_threshold"] = self.frozen_threshold
        res["architecture"] = self.model_type
        res["is_custom"] = not self.is_builtin
        res["version"] = self.model_version
        res["created_at"] = self.training_timestamp
        res["frozen"] = self.is_builtin
        return res


# Standard feature column lists for IBM AML Sequence models
IBM_AML_FEATURE_COLUMNS = [
    "amount_paid_log", "amount_received_log", "paid_received_diff_log",
    "fee_log", "fee_ratio", "is_free_transaction",
    "minute_of_day_sin", "minute_of_day_cos", "day_of_week_sin", "day_of_week_cos",
    "is_night", "is_weekend",
    "sender_hour_velocity_log", "sender_day_velocity_log", "sender_seconds_since_prior_log",
    "sender_prior_amount_paid_log", "sender_amount_ratio_vs_prior",
    "receiver_hour_velocity_log", "receiver_day_velocity_log", "receiver_seconds_since_prior_log",
    "receiver_prior_amount_received_log", "receiver_amount_ratio_vs_prior",
    "is_self_transfer", "is_inter_bank", "is_cross_currency",
    "from_bank", "to_bank", "receiving_currency", "payment_currency", "payment_format",
]
IBM_AML_CATEGORICAL_COLUMNS = ["from_bank", "to_bank", "receiving_currency", "payment_currency", "payment_format"]
IBM_AML_NUMERICAL_COLUMNS = [
    "amount_paid_log", "amount_received_log", "paid_received_diff_log",
    "fee_log", "fee_ratio", "is_free_transaction",
    "minute_of_day_sin", "minute_of_day_cos", "day_of_week_sin", "day_of_week_cos",
    "is_night", "is_weekend",
    "sender_hour_velocity_log", "sender_day_velocity_log", "sender_seconds_since_prior_log",
    "sender_prior_amount_paid_log", "sender_amount_ratio_vs_prior",
    "receiver_hour_velocity_log", "receiver_day_velocity_log", "receiver_seconds_since_prior_log",
    "receiver_prior_amount_received_log", "receiver_amount_ratio_vs_prior",
    "is_self_transfer", "is_inter_bank", "is_cross_currency",
]

# Built-in authoritative models
BUILTIN_MODELS: Dict[str, ModelSchemaMetadata] = {
    "HI-Small": ModelSchemaMetadata(
        model_id="HI-Small",
        model_name="HI-Small Champion (Federated LSTM FedAvg IID)",
        model_type="LSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-10-21T22:11:33",
        frozen_threshold=0.3377837836742401,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Lead federated LSTM sequence classifier trained on IBM AML HI-Small benchmark.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt"),
        expected_sha256="dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9555, "pr_auc": 0.4760, "f1_score": 0.4802},
    ),
    "hi_small_lstm_fedavg_iid_seed42": ModelSchemaMetadata(
        model_id="hi_small_lstm_fedavg_iid_seed42",
        model_name="Federated LSTM (FedAvg IID)",
        model_type="LSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-10-21T22:11:33",
        frozen_threshold=0.3377837836742401,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated LSTM sequence classifier trained on IBM AML HI-Small benchmark (FedAvg IID).",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt"),
        expected_sha256="dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9555, "pr_auc": 0.4760, "f1_score": 0.4802},
    ),
    "LI-Medium": ModelSchemaMetadata(
        model_id="LI-Medium",
        model_name="LI-Medium Champion (Federated BiLSTM FedAvg IID)",
        model_type="BiLSTM",
        dataset_name="LI-Medium",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-10-21T22:11:37",
        frozen_threshold=0.2384829968214035,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Lead federated BiLSTM sequence classifier trained on IBM AML LI-Medium benchmark.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/LI_Medium_Results/checkpoints/config_5/global_best.pt"),
        expected_sha256="bded77c75fcf89ea24af295fd7a5c28110744bf226a1c4c0a386f9924d1087d2",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/LI-Medium/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9709, "pr_auc": 0.3812, "f1_score": 0.4120},
    ),
    "centralized_lstm": ModelSchemaMetadata(
        model_id="centralized_lstm",
        model_name="Centralized Baseline (LSTM)",
        model_type="LSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-05T13:28:27",
        frozen_threshold=0.703287,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Centralized baseline LSTM model trained directly on pooled training transactions.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "models/checkpoints/centralized_lstm_seed42_20260905T132827/best.pt"),
        expected_sha256="382edf4e8ef82d9a00423901bc54fd957ee0821ecfc86d928609e7983c635afa",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9354, "pr_auc": 0.4771, "f1_score": 0.5529},
    ),
    "centralized_gru": ModelSchemaMetadata(
        model_id="centralized_gru",
        model_name="Centralized Baseline (GRU)",
        model_type="GRU",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-05T13:18:07",
        frozen_threshold=0.973227,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Centralized baseline GRU model trained directly on pooled training transactions.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "models/checkpoints/centralized_gru_seed42_20260905T131807/best.pt"),
        expected_sha256="a10e10f73a891db13771d9c8ec02a291e1c1daf231e460760e47a87f768e4095",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9328, "pr_auc": 0.2887, "f1_score": 0.3555},
    ),
    "centralized_bilstm": ModelSchemaMetadata(
        model_id="centralized_bilstm",
        model_name="Centralized Baseline (BiLSTM)",
        model_type="BiLSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-06T01:17:49",
        frozen_threshold=0.826391,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Centralized baseline BiLSTM model trained directly on pooled training transactions.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "models/checkpoints/centralized_bilstm_seed42_20260906T011749/best.pt"),
        expected_sha256="2b274cc93546a91c53275f423bb4af5819778cbfb38b9a23c85c805884ccfa49",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9245, "pr_auc": 0.4698, "f1_score": 0.5536},
    ),
    "hi_small_gru_fedavg_iid_seed42": ModelSchemaMetadata(
        model_id="hi_small_gru_fedavg_iid_seed42",
        model_name="Federated GRU (FedAvg IID)",
        model_type="GRU",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-18T12:37:51",
        frozen_threshold=0.500000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated GRU sequence classifier trained under IID client partitioning with FedAvg.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_gru_fedavg_iid_seed42/hi_small_gru_fedavg_iid_seed42_global_best.pt"),
        expected_sha256="1da9626d3ca24dadd93b6c758d76ede7dfae4b0f245fd4468552e2476b113c04",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9548, "pr_auc": 0.3950, "f1_score": 0.3401},
    ),
    "hi_small_bilstm_fedavg_iid_seed42": ModelSchemaMetadata(
        model_id="hi_small_bilstm_fedavg_iid_seed42",
        model_name="Federated BiLSTM (FedAvg IID)",
        model_type="BiLSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-18T13:10:00",
        frozen_threshold=0.500000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated BiLSTM sequence classifier trained under IID client partitioning with FedAvg.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_bilstm_fedavg_iid_seed42/hi_small_bilstm_fedavg_iid_seed42_global_best.pt"),
        expected_sha256="e9a8c43b09524e641d8d2f3a6d919076f589b7f77d2ab89695b66de30ba6c5aa",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9404, "pr_auc": 0.4078, "f1_score": 0.4438},
    ),
    "hi_small_lstm_fedprox_mu0.0001_iid_seed42": ModelSchemaMetadata(
        model_id="hi_small_lstm_fedprox_mu0.0001_iid_seed42",
        model_name="Federated LSTM (FedProx mu=0.0001 IID)",
        model_type="LSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-18T13:20:00",
        frozen_threshold=0.500000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated LSTM sequence classifier trained under IID client partitioning with FedProx (mu=0.0001).",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_lstm_fedprox_mu0.0001_iid_seed42/hi_small_lstm_fedprox_mu0.0001_iid_seed42_global_best.pt"),
        expected_sha256="d9ddeb275aa6b83ef67cf939ecc52f897e491c9ac706c535dfefd02f672a3f74",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9106, "pr_auc": 0.2824, "f1_score": 0.3079},
    ),
    "hi_small_gru_fedprox_mu0.0001_iid_seed42": ModelSchemaMetadata(
        model_id="hi_small_gru_fedprox_mu0.0001_iid_seed42",
        model_name="Federated GRU (FedProx mu=0.0001 IID)",
        model_type="GRU",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-18T13:30:00",
        frozen_threshold=0.500000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated GRU sequence classifier trained under IID client partitioning with FedProx (mu=0.0001).",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_gru_fedprox_mu0.0001_iid_seed42/hi_small_gru_fedprox_mu0.0001_iid_seed42_global_best.pt"),
        expected_sha256="fcd14fa426ed4ddbc6a00f1f5109286be0d7057648654c3c336fd6102f5cd52f",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9231, "pr_auc": 0.3382, "f1_score": 0.3450},
    ),
    "hi_small_bilstm_fedprox_mu0.0001_iid_seed42": ModelSchemaMetadata(
        model_id="hi_small_bilstm_fedprox_mu0.0001_iid_seed42",
        model_name="Federated BiLSTM (FedProx mu=0.0001 IID)",
        model_type="BiLSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-18T13:40:00",
        frozen_threshold=0.500000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated BiLSTM sequence classifier trained under IID client partitioning with FedProx (mu=0.0001).",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_bilstm_fedprox_mu0.0001_iid_seed42/hi_small_bilstm_fedprox_mu0.0001_iid_seed42_global_best.pt"),
        expected_sha256="ace199af13d8a86afbd16706582961ff9c5bc573e0fe8feaed76f716b75c7b8b",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9312, "pr_auc": 0.3421, "f1_score": 0.3702},
    ),
    "hi_small_lstm_fedavg_seed42": ModelSchemaMetadata(
        model_id="hi_small_lstm_fedavg_seed42",
        model_name="Federated LSTM (FedAvg Non-IID)",
        model_type="LSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-22T15:52:31",
        frozen_threshold=0.370000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated LSTM sequence classifier trained under Non-IID bank typologies with FedAvg.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_seed42/hi_small_lstm_fedavg_seed42_global_best.pt"),
        expected_sha256="1b64f79ddd4dc85d6bad953e93f2679bc102bf5fdbffeaf8b59df69ddd19ab1c",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9516, "pr_auc": 0.4505, "f1_score": 0.5029},
    ),
    "hi_small_gru_fedavg_seed42": ModelSchemaMetadata(
        model_id="hi_small_gru_fedavg_seed42",
        model_name="Federated GRU (FedAvg Non-IID)",
        model_type="GRU",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-22T15:55:00",
        frozen_threshold=0.700000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated GRU sequence classifier trained under Non-IID bank typologies with FedAvg.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_gru_fedavg_seed42/hi_small_gru_fedavg_seed42_global_best.pt"),
        expected_sha256="d225abce33d0dd350e5d9623aed77ec71a0591daf6d2d75cc77ff8d0c2394199",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9320, "pr_auc": 0.3150, "f1_score": 0.3680},
    ),
    "hi_small_bilstm_fedavg_seed42": ModelSchemaMetadata(
        model_id="hi_small_bilstm_fedavg_seed42",
        model_name="Federated BiLSTM (FedAvg Non-IID)",
        model_type="BiLSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-22T15:58:00",
        frozen_threshold=0.620000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated BiLSTM sequence classifier trained under Non-IID bank typologies with FedAvg.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_bilstm_fedavg_seed42/hi_small_bilstm_fedavg_seed42_global_best.pt"),
        expected_sha256="508427bde3560f2f1641040c8b2cfb0027fc37647b8724baba4eb24548d5bcf0",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9380, "pr_auc": 0.3120, "f1_score": 0.3850},
    ),
    "hi_small_lstm_fedprox_mu0.0001_seed42": ModelSchemaMetadata(
        model_id="hi_small_lstm_fedprox_mu0.0001_seed42",
        model_name="Federated LSTM (FedProx mu=0.0001 Non-IID)",
        model_type="LSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-22T16:00:00",
        frozen_threshold=0.360000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated LSTM sequence classifier trained under Non-IID bank typologies with FedProx (mu=0.0001).",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_lstm_fedprox_mu0.0001_seed42/hi_small_lstm_fedprox_mu0.0001_seed42_global_best.pt"),
        expected_sha256="35bf9029a3f1778f359c67ad7b54b27b7e92bab476d95a505a62914e383f26fd",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9079, "pr_auc": 0.2792, "f1_score": 0.3496},
    ),
    "hi_small_gru_fedprox_mu0.0001_seed42": ModelSchemaMetadata(
        model_id="hi_small_gru_fedprox_mu0.0001_seed42",
        model_name="Federated GRU (FedProx mu=0.0001 Non-IID)",
        model_type="GRU",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-22T16:05:00",
        frozen_threshold=0.690000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated GRU sequence classifier trained under Non-IID bank typologies with FedProx (mu=0.0001).",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_gru_fedprox_mu0.0001_seed42/hi_small_gru_fedprox_mu0.0001_seed42_global_best.pt"),
        expected_sha256="70cee297f8caec9719b0698447901b844e01b5506f3ec3439818603cc0f4eb58",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9150, "pr_auc": 0.3020, "f1_score": 0.3540},
    ),
    "hi_small_bilstm_fedprox_mu0.0001_seed42": ModelSchemaMetadata(
        model_id="hi_small_bilstm_fedprox_mu0.0001_seed42",
        model_name="Federated BiLSTM (FedProx mu=0.0001 Non-IID)",
        model_type="BiLSTM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-09-22T16:10:00",
        frozen_threshold=0.580000,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Federated BiLSTM sequence classifier trained under Non-IID bank typologies with FedProx (mu=0.0001).",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_bilstm_fedprox_mu0.0001_seed42/hi_small_bilstm_fedprox_mu0.0001_seed42_global_best.pt"),
        expected_sha256="17f1a3f01b7818c87ddc002024d15335af2b5c605a50cce7de5009d8159feb90",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9250, "pr_auc": 0.3080, "f1_score": 0.3620},
    ),
    "ensemble_hybrid": ModelSchemaMetadata(
        model_id="ensemble_hybrid",
        model_name="Validation-Optimized Hybrid Ensemble (alpha*=1.00)",
        model_type="Ensemble",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=IBM_AML_FEATURE_COLUMNS,
        categorical_columns=IBM_AML_CATEGORICAL_COLUMNS,
        numerical_columns=IBM_AML_NUMERICAL_COLUMNS,
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[10, 57],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-10-29T10:00:00",
        frozen_threshold=0.337784,
        is_sequence_model=True,
        required_sequence_fields=["timestamp", "from_account", "to_account", "amount"],
        description="Hybrid ensemble of IID and Non-IID champions with validation-optimal alpha*=1.00.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "results/HI_Small_Results/checkpoints/hi_small_lstm_fedavg_iid_seed42/hi_small_lstm_fedavg_iid_seed42_global_best.pt"),
        expected_sha256="dacfbb62e1e74a58a0c79237f7d0c16aa57df19253af82b8ab38af9bba2d46b4",
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9555, "pr_auc": 0.4760, "f1_score": 0.5359},
    ),
    "lightgbm_tabular_fallback": ModelSchemaMetadata(
        model_id="lightgbm_tabular_fallback",
        model_name="Fallback: no sequence history (LightGBM)",
        model_type="LightGBM",
        dataset_name="HI-Small",
        target_column="is_laundering",
        feature_columns=[
            "amount_paid_log", "amount_received_log", "amount_mismatch",
            "same_currency", "same_bank", "self_transfer", "payment_format_code",
            "payment_currency_code", "receiving_currency_code", "minute_of_day_sin",
            "minute_of_day_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
            "is_night", "sender_prior_count_log", "sender_prior_amount_log",
            "sender_seconds_since_prior_log", "sender_has_prior",
            "receiver_prior_count_log", "receiver_prior_amount_log",
            "receiver_seconds_since_prior_log", "receiver_has_prior",
        ],
        categorical_columns=["payment_format_code", "payment_currency_code", "receiving_currency_code"],
        numerical_columns=[
            "amount_paid_log", "amount_received_log", "amount_mismatch",
            "same_currency", "same_bank", "self_transfer", "minute_of_day_sin",
            "minute_of_day_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
            "is_night", "sender_prior_count_log", "sender_prior_amount_log",
            "sender_seconds_since_prior_log", "sender_has_prior",
            "receiver_prior_count_log", "receiver_prior_amount_log",
            "receiver_seconds_since_prior_log", "receiver_has_prior",
        ],
        preprocessing_version="v1.0",
        model_version="1.0.0",
        input_shape=[23],
        class_labels=["Legitimate", "Laundering"],
        training_timestamp="2026-11-03T08:56:00",
        frozen_threshold=0.337784,
        is_sequence_model=False,
        required_sequence_fields=[],
        description="Explicit tabular fallback LightGBM model for zero-history / single-transaction records.",
        is_builtin=True,
        checkpoint_path=str(BASE_DIR / "models/baselines/lightgbm_baseline.txt"),
        expected_sha256=None,
        pipeline_artifact_path=str(BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata/transformers.json"),
        training_metrics={"roc_auc": 0.9515, "pr_auc": 0.3678, "f1_score": 0.4039},
    ),
}


class UniversalModelRegistry:
    """Thread-safe registry for managing built-in and dynamically trained AML models."""

    _instance: Optional[UniversalModelRegistry] = None
    _lock = threading.Lock()

    def __new__(cls) -> UniversalModelRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._registry: Dict[str, ModelSchemaMetadata] = {}
                cls._instance._loaded_artifacts: Dict[str, Any] = {}
                cls._instance._initialize()
            return cls._instance

    def _initialize(self) -> None:
        """Initialize registry with built-in models and load existing dynamic models from disk."""
        # Register built-ins
        for k, v in BUILTIN_MODELS.items():
            self._registry[k] = v

        # Discover dynamic models from disk
        if DYNAMIC_MODELS_DIR.exists():
            for m_dir in DYNAMIC_MODELS_DIR.iterdir():
                if m_dir.is_dir():
                    meta_file = m_dir / "metadata.json"
                    if meta_file.exists():
                        try:
                            data = json.loads(meta_file.read_text(encoding="utf-8"))
                            meta = ModelSchemaMetadata(**data)
                            self._registry[meta.model_id] = meta
                            logger.info("Loaded dynamic model metadata for '%s'.", meta.model_id)
                        except Exception as e:
                            logger.warning("Failed to load metadata for dynamic model in '%s': %s", m_dir, e)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models in the registry."""
        with self._lock:
            return [m.to_dict() for m in self._registry.values()]

    def list_dynamic_models(self) -> List[ModelSchemaMetadata]:
        """List all registered custom-trained dynamic models."""
        with self._lock:
            return [m for m in self._registry.values() if not m.is_builtin]

    def get_model_metadata(self, model_id: str) -> Optional[ModelSchemaMetadata]:
        """Retrieve model metadata by identifier, handling aliases and case-insensitivity."""
        with self._lock:
            if model_id in self._registry:
                return self._registry[model_id]

            # Check aliases
            norm = model_id.lower().replace("-", "_").replace(" ", "_")
            if norm in ("auto", "best", "best_model", "champion"):
                return self._registry.get("HI-Small")
            if norm in ("ensemble", "hybrid_ensemble"):
                return self._registry.get("ensemble_hybrid") or self._registry.get("HI-Small")

            for k, meta in self._registry.items():
                if (
                    k.lower() == norm
                    or meta.model_id.lower() == norm
                    or meta.model_name.lower().replace(" ", "_") == norm
                    or meta.model_id.lower().replace("-", "_") == norm
                ):
                    return meta
            return None

    def register_dynamic_model(
        self,
        metadata: ModelSchemaMetadata,
        model_artifact: Any,
        preprocessor_artifact: Any,
    ) -> None:
        """Register a new user-trained dynamic model and persist its artifacts."""
        with self._lock:
            m_dir = DYNAMIC_MODELS_DIR / metadata.model_id
            m_dir.mkdir(parents=True, exist_ok=True)

            model_path = m_dir / "model.joblib"
            prep_path = m_dir / "preprocessor.joblib"
            meta_path = m_dir / "metadata.json"

            joblib.dump(model_artifact, model_path)
            joblib.dump(preprocessor_artifact, prep_path)

            metadata.checkpoint_path = str(model_path)
            metadata.pipeline_artifact_path = str(prep_path)
            metadata.is_builtin = False

            meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
            self._registry[metadata.model_id] = metadata
            self._loaded_artifacts[metadata.model_id] = {
                "model": model_artifact,
                "preprocessor": preprocessor_artifact,
            }
            logger.info("Successfully registered and persisted dynamic model '%s'.", metadata.model_id)

    def load_dynamic_model_artifacts(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load trained model object and fitted preprocessor for a dynamic model."""
        with self._lock:
            if model_id in self._loaded_artifacts:
                return self._loaded_artifacts[model_id]

            meta = self._registry.get(model_id)
            if not meta or meta.is_builtin or not meta.checkpoint_path:
                return None

            model_path = Path(meta.checkpoint_path)
            prep_path = Path(meta.pipeline_artifact_path) if meta.pipeline_artifact_path else None

            if not model_path.exists():
                return None

            model = joblib.load(model_path)
            preprocessor = joblib.load(prep_path) if (prep_path and prep_path.exists()) else None

            artifacts = {"model": model, "preprocessor": preprocessor}
            self._loaded_artifacts[model_id] = artifacts
            return artifacts


