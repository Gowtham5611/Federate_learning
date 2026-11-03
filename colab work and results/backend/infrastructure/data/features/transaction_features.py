"""Per-transaction features: amounts, and the relationships between the two sides.

Everything here is computed from a single row. No state, no ordering
requirement, no possibility of leakage — which is why this builder is the
simplest one in the package and why its features are the ones that survive
unchanged into every later phase.

Three relationships in the verified data motivated these features, and the
measured frequencies are recorded because they are what make each feature worth
a column:

* ``Amount Received == Amount Paid`` in 5,006,187 rows (98.5791%), so a mismatch
  marks the unusual 1.42%
* ``Receiving Currency == Payment Currency`` in 5,006,175 rows (98.5789%)
* sender and receiver accounts are identical in 591,212 rows (11.6418%)

Amounts are log-transformed rather than clipped. The verified minimum is 1e-06
and the maximum is 1.0463e+12 — twelve orders of magnitude — and the largest
values are not errors to be trimmed away but the transactions an AML system
most needs to see. ``log1p`` compresses that range while keeping every value.
"""

from __future__ import annotations

import numpy as np
from pandas import DataFrame

from infrastructure.data.preprocessors.base_preprocessor import FeatureBuilder
from infrastructure.data.schemas.ibm_aml_schema import (
    COL_AMOUNT_PAID,
    COL_AMOUNT_RECEIVED,
    COL_FROM_ACCOUNT,
    COL_FROM_BANK,
    COL_PAYMENT_CURRENCY,
    COL_RECEIVING_CURRENCY,
    COL_TO_ACCOUNT,
    COL_TO_BANK,
)

LOG_TRANSFORMED_COLUMNS: tuple[str, ...] = (
    "amount_paid_log",
    "amount_received_log",
)
"""Columns this builder emits already log-transformed, for the scaler to scale."""


class TransactionFeatureBuilder(FeatureBuilder):
    """Builds the row-local transaction features."""

    @property
    def produced_columns(self) -> tuple[str, ...]:
        """The six transaction features, in output order."""
        return (
            "amount_paid_log",
            "amount_received_log",
            "amount_mismatch",
            "same_currency",
            "same_bank",
            "self_transfer",
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The raw columns each feature is derived from."""
        return (
            COL_AMOUNT_PAID,
            COL_AMOUNT_RECEIVED,
            COL_PAYMENT_CURRENCY,
            COL_RECEIVING_CURRENCY,
            COL_FROM_BANK,
            COL_TO_BANK,
            COL_FROM_ACCOUNT,
            COL_TO_ACCOUNT,
        )

    @property
    def rejected_features(self) -> dict[str, str]:
        """Transaction features considered and not built, with the reason."""
        return {
            "amount_difference": (
                "'Amount Received' and 'Amount Paid' are denominated in "
                "different currencies in 1.42% of rows, so their arithmetic "
                "difference is not a quantity. No exchange-rate table exists in "
                "the dataset and inventing one would fabricate data. The "
                "boolean 'amount_mismatch' captures the same signal without "
                "asserting a conversion."
            ),
            "amount_in_common_currency": (
                "Would require exchange rates that the dataset does not "
                "contain. Rates for 2022-09 could be fetched externally, but "
                "that would silently introduce data from outside the dataset "
                "into a preprocessing step."
            ),
            "bank_identifier_encoded": (
                "356,303 distinct bank identifiers. See "
                "HIGH_CARDINALITY_REFUSALS in the categorical encoder: an "
                "integer bank id has no magnitude, so it is neither scaled nor "
                "encoded. 'same_bank' keeps the part that carries signal."
            ),
        }

    def apply(self, chunk: DataFrame) -> DataFrame:
        """Add the transaction features to a copy of ``chunk``."""
        result = chunk.copy()

        paid = result[COL_AMOUNT_PAID].astype("float64")
        received = result[COL_AMOUNT_RECEIVED].astype("float64")

        # Safe without clipping: the verified minimum across all 5,078,345 rows
        # is 1e-06, and no non-positive amount exists in either column.
        result["amount_paid_log"] = np.log1p(paid.clip(lower=0.0)).astype("float32")
        result["amount_received_log"] = np.log1p(received.clip(lower=0.0)).astype("float32")

        result["amount_mismatch"] = (paid != received).astype("int8")
        result["same_currency"] = (
            result[COL_PAYMENT_CURRENCY].astype(str)
            == result[COL_RECEIVING_CURRENCY].astype(str)
        ).astype("int8")
        result["same_bank"] = (
            result[COL_FROM_BANK].astype(str) == result[COL_TO_BANK].astype(str)
        ).astype("int8")
        result["self_transfer"] = (
            result[COL_FROM_ACCOUNT].astype(str) == result[COL_TO_ACCOUNT].astype(str)
        ).astype("int8")
        return result
