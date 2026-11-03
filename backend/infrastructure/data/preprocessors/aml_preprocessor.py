"""The IBM AML preprocessor: raw verified columns in, model-ready columns out.

This class owns the *per-chunk* transformation and the fitted state. It does not
own the passes over the file, the interim partitions or the outputs — those
belong to
:class:`~application.services.preprocessing_service.PreprocessingService`,
which orchestrates them. The split exists so that the leakage-critical logic
here can be tested on a handful of rows without any files at all.

Three rules are enforced structurally rather than by convention:

*Nothing is fitted outside training.* :meth:`fit_on_training_chunk` refuses to
run after :meth:`freeze_fit`, and :meth:`transform` refuses to run before it.
There is no ordering of calls that fits on validation or test data.

*Nothing is dropped silently.* :meth:`clean` returns the surviving rows together
with a count for every rule that removed anything, and the caller accumulates
them into the ledger that the final report prints.

*Nothing looks forward.* The only stateful builder accumulates strictly
backwards-looking history; see
:mod:`infrastructure.data.features.account_history`.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from domain.exceptions.preprocessing import TransformerNotFittedError
from infrastructure.data.features.account_history import (
    LOG_TRANSFORMED_COLUMNS as HISTORY_LOG_COLUMNS,
)
from infrastructure.data.features.account_history import AccountHistoryBuilder
from infrastructure.data.features.temporal_features import (
    EPOCH_SECONDS_COLUMN,
    PARSED_TIMESTAMP_COLUMN,
    TemporalFeatureBuilder,
    parse_timestamps,
)
from infrastructure.data.features.transaction_features import (
    LOG_TRANSFORMED_COLUMNS as TRANSACTION_LOG_COLUMNS,
)
from infrastructure.data.features.transaction_features import TransactionFeatureBuilder
from infrastructure.data.preprocessors.categorical_encoder import (
    HIGH_CARDINALITY_REFUSALS,
    OrdinalCategoricalEncoder,
)
from infrastructure.data.preprocessors.running_stats import (
    RunningStandardizer,
    describe_log_transform,
)
from infrastructure.data.schemas.ibm_aml_schema import (
    COL_AMOUNT_PAID,
    COL_AMOUNT_RECEIVED,
    COL_FROM_ACCOUNT,
    COL_FROM_BANK,
    COL_IS_LAUNDERING,
    COL_PAYMENT_CURRENCY,
    COL_PAYMENT_FORMAT,
    COL_RECEIVING_CURRENCY,
    COL_TIMESTAMP,
    COL_TO_ACCOUNT,
    COL_TO_BANK,
    RAW_COLUMN_ORDER,
    TIMESTAMP_FORMAT,
)

logger = logging.getLogger(__name__)

PREPROCESSING_VERSION = "1.0.0"
"""Stamped into every artefact. Bump when the output columns or their meaning change."""

OUTPUT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
"""Timestamp format in the processed output and in every reported boundary.

Fixed-width and zero-padded, so lexicographic comparison is chronological
comparison — which is what lets the splitter compare boundaries as text. The raw
file's ``YYYY/MM/DD HH:MM`` has the same property, but one format throughout is
worth more than fidelity to the input's punctuation.
"""

RAW_DTYPES: dict[str, str] = {
    COL_TIMESTAMP: "string",
    COL_FROM_BANK: "string",
    COL_FROM_ACCOUNT: "string",
    COL_TO_BANK: "string",
    COL_TO_ACCOUNT: "string",
    COL_RECEIVING_CURRENCY: "string",
    COL_PAYMENT_CURRENCY: "string",
    COL_PAYMENT_FORMAT: "string",
}
"""Types forced on read, for the columns where inference is unsafe.

Identifiers are read as text even though the verified bank columns hold
integers. An identifier that is inferred as ``int64`` in one chunk and as text
in another compares unequal to itself across the boundary, which would corrupt
both the ``same_bank`` feature and the per-account history. Amount and target
columns are left to inference because they *are* numbers and the cleaning step
coerces them explicitly.
"""

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    COL_PAYMENT_FORMAT,
    COL_PAYMENT_CURRENCY,
    COL_RECEIVING_CURRENCY,
)
"""The three closed-vocabulary columns that are encoded. See the encoder for the
identifier columns that are deliberately not."""

SCALED_COLUMNS: tuple[str, ...] = TRANSACTION_LOG_COLUMNS + HISTORY_LOG_COLUMNS
"""Columns standardised after their ``log1p``.

Only the skewed magnitudes. The cyclical time features are already bounded in
``[-1, 1]``, the boolean flags are already ``{0, 1}``, and the categorical codes
are nominal — standardising any of those would either destroy the encoding or
achieve nothing.
"""

TARGET_OUTPUT_COLUMN = "is_laundering"

TARGET_MAPPING: dict[str, int] = {"0": 0, "1": 1}
"""Label mapping, taken from the two values verified to occur in the file.

Not an assumption about what the column *should* contain: a full chunked pass
over all 5,078,345 rows found exactly ``0`` and ``1``, with counts 5,073,168 and
5,177. Anything else is treated as an invalid row and counted.
"""

POSITIVE_LABEL = 1

IDENTIFIER_COLUMNS: tuple[str, ...] = ("timestamp", "from_account", "to_account")
"""Carried through the output but never model inputs.

They exist so the next phase can order transactions in time and group them per
account to build sequences, which an LSTM or GRU needs and which cannot be
reconstructed once identity is dropped. The feature metadata marks them as
identifiers so they are not fed to a model by accident.
"""

_OUTPUT_RENAMES: dict[str, str] = {
    COL_TIMESTAMP: "timestamp",
    COL_FROM_ACCOUNT: "from_account",
    COL_TO_ACCOUNT: "to_account",
    COL_IS_LAUNDERING: TARGET_OUTPUT_COLUMN,
    f"{COL_PAYMENT_FORMAT}_code": "payment_format_code",
    f"{COL_PAYMENT_CURRENCY}_code": "payment_currency_code",
    f"{COL_RECEIVING_CURRENCY}_code": "receiving_currency_code",
}
"""Raw-to-output names.

This is where the duplicated ``Account`` header finally stops being ambiguous:
what pandas calls ``Account`` and ``Account.1`` becomes ``from_account`` and
``to_account``, matching the ``From Bank`` / ``To Bank`` pair they travel with.
"""

CATEGORICAL_CODE_COLUMNS: tuple[str, ...] = (
    "payment_format_code",
    "payment_currency_code",
    "receiving_currency_code",
)


class AmlPreprocessor:
    """Cleans, engineers, fits and transforms IBM AML transaction chunks."""

    def __init__(self, history_builder: Optional[AccountHistoryBuilder] = None) -> None:
        """Build the feature pipeline in an unfitted state."""
        self._temporal = TemporalFeatureBuilder()
        self._transaction = TransactionFeatureBuilder()
        self._history = history_builder if history_builder is not None else AccountHistoryBuilder()
        self._encoder = OrdinalCategoricalEncoder(CATEGORICAL_COLUMNS)
        self._standardizer = RunningStandardizer(SCALED_COLUMNS)
        self._unknown_categories: Counter[str] = Counter()

    @property
    def history_builder(self) -> AccountHistoryBuilder:
        """Access the underlying account history builder."""
        return self._history

    # --- Properties describing the output ------------------------------------

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Model-input columns, in output order."""
        return (
            *self._transaction.produced_columns,
            *CATEGORICAL_CODE_COLUMNS,
            *self._temporal.produced_columns,
            *self._history.produced_columns,
        )

    @property
    def identifier_columns(self) -> tuple[str, ...]:
        """Non-feature columns carried for traceability and sequencing."""
        return IDENTIFIER_COLUMNS

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Every column written to the processed files, in order."""
        return (*self.identifier_columns, *self.feature_columns, TARGET_OUTPUT_COLUMN)

    @property
    def engineered_columns(self) -> tuple[str, ...]:
        """Columns that must survive between the engineering and transform passes.

        Deliberately narrower than "everything :meth:`engineer` produced". The
        parsed-timestamp and epoch columns have done their work by then, and the
        raw bank and amount columns have been folded into features, so carrying
        them into the interim staging file would cost disk for nothing. What
        remains is what :meth:`transform` reads plus what the output keeps.
        """
        return (
            COL_TIMESTAMP,
            COL_FROM_ACCOUNT,
            COL_TO_ACCOUNT,
            *CATEGORICAL_COLUMNS,
            *self._transaction.produced_columns,
            *self._temporal.produced_columns,
            *self._history.produced_columns,
            COL_IS_LAUNDERING,
        )

    @property
    def is_fitted(self) -> bool:
        """Whether fitting has been closed and transforming is allowed."""
        return self._encoder.is_fitted and self._standardizer.is_fitted

    @property
    def distinct_accounts(self) -> int:
        """Accounts tracked by the history state, which is what bounds memory."""
        return self._history.distinct_accounts

    @property
    def unknown_category_counts(self) -> dict[str, int]:
        """Values seen outside the training vocabulary, per column."""
        return dict(self._unknown_categories)

    # --- Stage 1: cleaning ---------------------------------------------------

    def clean(self, chunk: DataFrame) -> tuple[DataFrame, Counter[str]]:
        """Normalise types, derive the parsed timestamp, and drop invalid rows.

        Every drop is counted. The removal keys match the ledger entries in
        :func:`removal_descriptions`, so the report cannot describe a rule the
        code does not apply.

        Args:
            chunk: Raw rows carrying the verified column names.

        Returns:
            The surviving rows, and a count per removal reason.
        """
        frame = chunk.copy()
        removed: Counter[str] = Counter()
        before = len(frame)

        for column in (COL_AMOUNT_PAID, COL_AMOUNT_RECEIVED, COL_IS_LAUNDERING):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        present = frame[list(RAW_COLUMN_ORDER)].notna().all(axis=1)
        removed["missing_required_value"] = int((~present).sum())
        frame = frame[present]

        parsed = parse_timestamps(frame[COL_TIMESTAMP], TIMESTAMP_FORMAT)
        parseable = parsed.notna()
        removed["unparseable_timestamp"] = int((~parseable).sum())
        frame = frame[parseable]
        parsed = parsed[parseable]

        non_negative = (frame[COL_AMOUNT_PAID] >= 0) & (frame[COL_AMOUNT_RECEIVED] >= 0)
        removed["negative_amount"] = int((~non_negative).sum())
        frame = frame[non_negative]
        parsed = parsed[non_negative]

        known_label = (
            frame[COL_IS_LAUNDERING].astype("int64").astype(str).isin(TARGET_MAPPING)
        )
        removed["unknown_target_value"] = int((~known_label).sum())
        frame = frame[known_label]
        parsed = parsed[known_label]

        frame = frame.copy()
        frame[PARSED_TIMESTAMP_COLUMN] = parsed
        # Rewriting the timestamp in the canonical output format here means the
        # histogram, the boundaries, the split reports and the output files all
        # compare the same strings.
        frame[COL_TIMESTAMP] = parsed.dt.strftime(OUTPUT_TIMESTAMP_FORMAT)
        frame[EPOCH_SECONDS_COLUMN] = (
            parsed - pd.Timestamp("1970-01-01")
        ).dt.total_seconds()
        frame[COL_IS_LAUNDERING] = frame[COL_IS_LAUNDERING].astype("int8")

        if before != len(frame):
            logger.warning(
                "Cleaning removed %d of %d row(s): %s",
                before - len(frame),
                before,
                dict(removed),
            )
        return frame, removed

    @staticmethod
    def day_keys(frame: DataFrame) -> Series:
        """The calendar date of each row, as ``YYYY-MM-DD``.

        Taken from the already-canonical timestamp text rather than re-derived
        from the parsed datetime, so a partition key can never disagree with the
        timestamp written beside it.
        """
        return frame[COL_TIMESTAMP].str[:10]

    @staticmethod
    def prepare_partition(frame: DataFrame) -> DataFrame:
        """Re-derive the working columns after a partition is read back.

        The partitions store only the eleven verified columns, with the
        timestamp already in canonical form. Parsing it again here is cheaper
        than having written two extra columns for every one of five million
        rows, and it keeps the interim files readable as ordinary CSV.
        """
        prepared = frame.copy()
        parsed = parse_timestamps(prepared[COL_TIMESTAMP], OUTPUT_TIMESTAMP_FORMAT)
        prepared[PARSED_TIMESTAMP_COLUMN] = parsed
        prepared[EPOCH_SECONDS_COLUMN] = (
            parsed - pd.Timestamp("1970-01-01")
        ).dt.total_seconds()
        for column in (COL_AMOUNT_PAID, COL_AMOUNT_RECEIVED):
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[COL_IS_LAUNDERING] = pd.to_numeric(
            prepared[COL_IS_LAUNDERING], errors="coerce"
        ).astype("int8")
        return prepared

    # --- Stage 2: feature engineering ----------------------------------------

    def engineer(self, chunk: DataFrame) -> DataFrame:
        """Add every derived feature to a chronologically ordered chunk.

        Args:
            chunk: Cleaned rows **in canonical chronological order**. Passing
                unordered rows produces wrong history features rather than an
                error, which is why the service sorts before calling this and
                why this method is not part of the public pipeline surface.

        Returns:
            A new frame with the temporal, transaction and behavioural columns.
        """
        frame = self._temporal.apply(chunk)
        frame = self._transaction.apply(frame)
        return self._history.apply(frame)

    @staticmethod
    def sort_chronologically(frame: DataFrame) -> DataFrame:
        """Put a frame into the canonical order the pipeline defines.

        The dataset's resolution is one minute, so many rows share a timestamp
        and "chronological order" alone does not determine a unique sequence.
        The canonical order is *timestamp, then position in the raw file*, with
        a stable sort. Two consequences matter:

        * It is deterministic, so two runs produce identical history features.
        * A row still never sees a row from a later minute, so no future
          information enters. Within a single minute the ordering is a
          convention, and it is this one.
        """
        return frame.sort_values(
            by=[COL_TIMESTAMP], kind="stable", ignore_index=True
        )

    # --- Stage 3: fit, then transform ----------------------------------------

    def fit_on_training_chunk(self, chunk: DataFrame) -> None:
        """Learn category vocabularies and scaling statistics from training rows.

        Args:
            chunk: Engineered rows belonging to the training split **only**.

        Raises:
            TransformerAlreadyFittedError: :meth:`freeze_fit` was already called.
        """
        self._encoder.update(chunk)
        self._standardizer.update(chunk)

    def freeze_fit(self) -> None:
        """Close fitting. Idempotent, and required before :meth:`transform`."""
        self._encoder.freeze()
        self._standardizer.freeze()

    def transform(self, chunk: DataFrame) -> DataFrame:
        """Encode, scale and select the output columns.

        Args:
            chunk: Engineered rows from any split.

        Returns:
            A frame containing exactly :attr:`output_columns`, in that order.

        Raises:
            TransformerNotFittedError: :meth:`freeze_fit` has not been called.
        """
        if not self.is_fitted:
            raise TransformerNotFittedError("AmlPreprocessor")

        self._unknown_categories.update(self._encoder.count_unknown(chunk))

        frame = self._encoder.transform(chunk)
        frame = self._standardizer.transform(frame)
        frame = frame.rename(columns=_OUTPUT_RENAMES)
        return frame[list(self.output_columns)]

    # --- Reporting -----------------------------------------------------------

    def target_mapping(self) -> dict[str, int]:
        """The label mapping actually applied."""
        return dict(TARGET_MAPPING)

    def describe(self) -> dict[str, Any]:
        """Full preprocessing metadata for the artefacts.

        Raises:
            TransformerNotFittedError: Fitting has not been closed, so the
                learned parameters do not exist yet.
        """
        if not self.is_fitted:
            raise TransformerNotFittedError("AmlPreprocessor", "describe")

        scaling = self._standardizer.parameters()
        return {
            "preprocessing_version": PREPROCESSING_VERSION,
            "fitted_on": "train split only",
            "timestamp_format": OUTPUT_TIMESTAMP_FORMAT,
            "column_order": list(self.output_columns),
            "identifier_columns": list(self.identifier_columns),
            "feature_columns": list(self.feature_columns),
            "target_column": TARGET_OUTPUT_COLUMN,
            "raw_to_output_names": dict(_OUTPUT_RENAMES),
            "categorical_encoding": self._encoder.describe(),
            "numerical_transformations": {
                **describe_log_transform(SCALED_COLUMNS),
                **{
                    column: {
                        "transform": "none",
                        "reason": (
                            "already bounded in [-1, 1] by the cyclical encoding"
                        ),
                    }
                    for column in self._temporal.produced_columns
                    if column.endswith(("_sin", "_cos"))
                },
                **{
                    column: {
                        "transform": "none",
                        "reason": "boolean indicator, already in {0, 1}",
                    }
                    for column in (
                        "amount_mismatch",
                        "same_currency",
                        "same_bank",
                        "self_transfer",
                        "is_weekend",
                        "is_night",
                        "sender_has_prior",
                        "receiver_has_prior",
                    )
                },
                **{
                    column: {
                        "transform": "none",
                        "reason": (
                            "nominal category code; scaling would imply an "
                            "ordering the values do not have"
                        ),
                    }
                    for column in CATEGORICAL_CODE_COLUMNS
                },
            },
            "scaling_parameters": scaling,
            "scaled_columns": list(SCALED_COLUMNS),
            "unscaled_columns": [
                column for column in self.feature_columns if column not in SCALED_COLUMNS
            ],
            "columns_not_encoded": dict(HIGH_CARDINALITY_REFUSALS),
            "feature_builders": [
                self._transaction.describe(),
                self._temporal.describe(),
                self._history.describe(),
            ],
            "unknown_category_counts": dict(self._unknown_categories),
        }

    def rejected_features(self) -> dict[str, str]:
        """Every feature deliberately not built, across all builders."""
        rejected: dict[str, str] = {}
        for builder in (self._transaction, self._temporal, self._history):
            rejected.update(builder.rejected_features)
        return rejected


def removal_descriptions() -> dict[str, str]:
    """What each cleaning rule tests, keyed by the counter it increments."""
    return {
        "missing_required_value": (
            "any of the 11 verified columns was null. The verification report "
            "measured zero nulls across the whole file, so a non-zero count "
            "here means the input differs from the file that was verified."
        ),
        "unparseable_timestamp": (
            f"the timestamp did not match the verified format "
            f"'{TIMESTAMP_FORMAT}'. A row with no usable time cannot be placed "
            f"in a chronological split at all."
        ),
        "negative_amount": (
            "an amount was below zero. Large amounts are kept - unusual values "
            "carry signal in AML and are never treated as outliers to remove - "
            "but a negative transferred amount is not a magnitude."
        ),
        "unknown_target_value": (
            "the label was neither 0 nor 1, the only two values found in the "
            "verified file. Such a row has no ground truth to learn from."
        ),
    }


def count_duplicate_rows(frame: DataFrame) -> int:
    """Count exact duplicate rows within one day's partition.

    Exact enough to be complete: two identical rows carry identical timestamps
    and therefore land in the same day partition, so counting per day counts
    every duplicate in the dataset without ever holding more than one day.

    Duplicates are **reported and kept**. At one-minute resolution, two
    transactions of the same amount between the same accounts in the same minute
    are entirely plausible, and deleting genuine transactions from an AML
    dataset is a worse error than keeping a repeat.
    """
    return int(frame.duplicated(subset=list(RAW_COLUMN_ORDER)).sum())


def summarise_class_counts(labels: Series) -> dict[int, int]:
    """Count rows per encoded label."""
    counts = labels.astype("int64").value_counts()
    return {int(str(label)): int(count) for label, count in counts.items()}


def to_float(value: object) -> float:
    """Coerce a numpy scalar to a plain float for JSON serialisation."""
    return float(np.asarray(value).item())
