"""Per-account behavioural history, accumulated strictly forwards in time.

This is the only stateful builder in the package, and the only one where
leakage is a live risk, so the causality rule is enforced by the shape of the
computation rather than by care:

    the features for row *i* are derived from rows *0 … i-1* only,
    in the canonical chronological order, and the state is updated
    **after** those features have been read.

Concretely, "how many transactions had this account sent before this one" is
computed as ``state[account] + (position of this row within its account group in
this chunk)``, and only then is ``state[account]`` advanced. A row can therefore
never see itself or anything after it. Because the caller feeds chunks in
chronological order, the result is identical to what a single in-memory pass
would produce — which is the property that makes chunking safe rather than
merely convenient.

**State across splits is not leakage.** The accumulator keeps running from the
training days into validation and test. That is deliberate and is *not* the
thing Rule 4 prohibits: a validation transaction's account history is
information that existed before that transaction happened, and a deployed system
would have it. What must not cross a split boundary is a *fitted statistic* — a
mean, a standard deviation, a category vocabulary — and none of those are
computed here. Resetting history at each boundary would instead fabricate a
world in which every account appears brand new on day 7, which is both wrong and
optimistic.

**Memory.** State is proportional to *distinct accounts*, not rows: six float64
arrays plus one string-keyed index. The verified file contains 515,080 distinct
accounts, so the arrays total about 25 MB and the index dominates at roughly
60 MB — bounded, and independent of the five million rows streaming past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from infrastructure.data.features.temporal_features import EPOCH_SECONDS_COLUMN
from infrastructure.data.preprocessors.base_preprocessor import FeatureBuilder
from infrastructure.data.schemas.ibm_aml_schema import (
    COL_AMOUNT_PAID,
    COL_AMOUNT_RECEIVED,
    COL_FROM_ACCOUNT,
    COL_TO_ACCOUNT,
)

_INITIAL_CAPACITY = 1024

LOG_TRANSFORMED_COLUMNS: tuple[str, ...] = (
    "sender_prior_count_log",
    "sender_prior_amount_log",
    "sender_seconds_since_prior_log",
    "receiver_prior_count_log",
    "receiver_prior_amount_log",
    "receiver_seconds_since_prior_log",
)
"""Columns this builder emits already log-transformed, for the scaler to scale."""

NO_PRIOR_SECONDS = 0.0
"""Elapsed-time value used when an account has no earlier transaction.

Paired with a ``*_has_prior`` flag rather than left as a missing value or a
negative sentinel: zero keeps the column numeric and finite, and the flag tells
a model that the zero means "never seen" rather than "seen a moment ago".
"""


class _SideState:
    """The three running arrays for one side of the transaction.

    Held in an object rather than as six arrays on the builder because the
    arrays are *replaced* when they grow, not resized in place. A caller that
    had read ``self._sent_count`` into a local before the growth would go on
    writing to the discarded array, so every reader reaches the arrays through
    this object and always sees the current one.
    """

    def __init__(self, capacity: int) -> None:
        """Start with no history for any account."""
        self.count = np.zeros(capacity, dtype=np.float64)
        self.total = np.zeros(capacity, dtype=np.float64)
        self.last = np.full(capacity, np.nan, dtype=np.float64)

    def grow(self, capacity: int) -> None:
        """Replace the arrays with larger ones holding the same history."""
        self.count = _grown(self.count, capacity, 0.0)
        self.total = _grown(self.total, capacity, 0.0)
        self.last = _grown(self.last, capacity, np.nan)


class AccountHistoryBuilder(FeatureBuilder):
    """Accumulates sending and receiving history per account, chunk by chunk.

    One instance must process the entire dataset in chronological order. Reusing
    an instance across two unrelated datasets, or feeding it chunks out of
    order, silently produces wrong features; the pipeline owns exactly one
    instance for exactly one run.
    """

    def __init__(self) -> None:
        """Start with empty history."""
        self._index: dict[str, int] = {}
        self._capacity = _INITIAL_CAPACITY
        self._sent = _SideState(self._capacity)
        self._received = _SideState(self._capacity)

    @property
    def distinct_accounts(self) -> int:
        """How many distinct accounts have been seen so far."""
        return len(self._index)

    @property
    def produced_columns(self) -> tuple[str, ...]:
        """The eight behavioural features, in output order."""
        return (
            "sender_prior_count_log",
            "sender_prior_amount_log",
            "sender_seconds_since_prior_log",
            "sender_has_prior",
            "receiver_prior_count_log",
            "receiver_prior_amount_log",
            "receiver_seconds_since_prior_log",
            "receiver_has_prior",
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The raw columns and the derived epoch column this builder consumes."""
        return (
            COL_FROM_ACCOUNT,
            COL_TO_ACCOUNT,
            COL_AMOUNT_PAID,
            COL_AMOUNT_RECEIVED,
            EPOCH_SECONDS_COLUMN,
        )

    @property
    def rejected_features(self) -> dict[str, str]:
        """Behavioural features considered and not built, with the reason."""
        return {
            "counterparty_pair_prior_count": (
                "Would need state keyed by (sender, receiver) pair. The number "
                "of distinct pairs is unbounded by anything measured and can "
                "approach the row count, so the state would grow with the "
                "dataset rather than with the account population - the exact "
                "property that keeps this builder memory-safe."
            ),
            "account_total_transaction_count": (
                "A whole-dataset count per account uses transactions that "
                "happen after the row it would describe. That is future "
                "information; the causal 'prior count' features replace it."
            ),
            "account_degree_over_window": (
                "A rolling distinct-counterparty count needs a per-account "
                "window buffer, making state grow with window length times "
                "account count. Deferred rather than rejected outright, but not "
                "implemented in this phase."
            ),
            "is_fraud_ring_member": (
                "Any feature derived from the label leaks the target directly. "
                "Not implemented under any windowing scheme."
            ),
        }

    def apply(self, chunk: DataFrame) -> DataFrame:
        """Add the behavioural features and advance the history state.

        Args:
            chunk: Rows in canonical chronological order, carrying
                :attr:`required_columns`.

        Returns:
            A new frame with the eight behavioural columns added.
        """
        result = chunk.copy()
        epoch = pd.to_numeric(result[EPOCH_SECONDS_COLUMN], errors="coerce").astype("float64")

        sender = self._side(
            result[COL_FROM_ACCOUNT],
            pd.to_numeric(result[COL_AMOUNT_PAID], errors="coerce").astype("float64"),
            epoch,
            self._sent,
        )
        receiver = self._side(
            result[COL_TO_ACCOUNT],
            pd.to_numeric(result[COL_AMOUNT_RECEIVED], errors="coerce").astype("float64"),
            epoch,
            self._received,
        )

        for prefix, (count, amount, seconds, has_prior) in (
            ("sender", sender),
            ("receiver", receiver),
        ):
            result[f"{prefix}_prior_count_log"] = np.log1p(count).astype("float32")
            result[f"{prefix}_prior_amount_log"] = np.log1p(amount).astype("float32")
            result[f"{prefix}_seconds_since_prior_log"] = np.log1p(seconds).astype("float32")
            result[f"{prefix}_has_prior"] = has_prior.astype("int8")

        return result

    def _side(
        self,
        accounts: Series,
        amounts: Series,
        epoch: Series,
        state: _SideState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute one side's prior-history features, then advance its state.

        Returns:
            ``(prior_count, prior_amount, seconds_since_prior, has_prior)``.
        """
        codes = self._codes_for(accounts)
        key = pd.Series(codes, index=accounts.index)

        # Exclusive prefixes within this chunk: cumcount is 0 for the first row
        # of each group, and subtracting the row's own amount from the cumulative
        # sum leaves the sum of the rows before it.
        within_count = key.groupby(key, sort=False).cumcount().to_numpy(dtype=np.float64)
        within_sum = (amounts.groupby(key, sort=False).cumsum() - amounts).to_numpy(
            dtype=np.float64
        )
        previous_in_chunk = (
            epoch.groupby(key, sort=False).shift(1).to_numpy(dtype=np.float64)
        )

        # Read through ``state`` rather than into locals: ``_codes_for`` above
        # may just have replaced these arrays with larger ones.
        prior_count = state.count[codes] + within_count
        prior_amount = state.total[codes] + within_sum

        # A row that is the first of its account in this chunk falls back to the
        # last time that account was seen in an *earlier* chunk; NaN there means
        # the account has genuinely never appeared before.
        previous = np.where(
            np.isnan(previous_in_chunk), state.last[codes], previous_in_chunk
        )
        has_prior = ~np.isnan(previous)
        elapsed = np.where(
            has_prior, epoch.to_numpy(dtype=np.float64) - previous, NO_PRIOR_SECONDS
        )
        seconds_since_prior = np.maximum(np.nan_to_num(elapsed, nan=0.0), 0.0)

        self._advance(codes, amounts, epoch, state)
        return prior_count, prior_amount, seconds_since_prior, has_prior

    @staticmethod
    def _advance(
        codes: np.ndarray,
        amounts: Series,
        epoch: Series,
        state: _SideState,
    ) -> None:
        """Fold this chunk's rows into the running per-account state."""
        folded = pd.DataFrame(
            {
                "code": codes,
                "amount": amounts.to_numpy(dtype=np.float64),
                "epoch": epoch.to_numpy(dtype=np.float64),
            }
        )
        grouped = folded.groupby("code", sort=False)
        aggregated = grouped.agg(
            rows=("amount", "size"), total=("amount", "sum"), latest=("epoch", "max")
        )
        positions = aggregated.index.to_numpy(dtype=np.int64)

        state.count[positions] += aggregated["rows"].to_numpy(dtype=np.float64)
        state.total[positions] += np.nan_to_num(
            aggregated["total"].to_numpy(dtype=np.float64), nan=0.0
        )
        # Chunks arrive in chronological order, so this chunk's latest timestamp
        # for an account is never earlier than the one already stored.
        latest = aggregated["latest"].to_numpy(dtype=np.float64)
        stored = state.last[positions]
        state.last[positions] = np.where(np.isnan(stored), latest, np.maximum(stored, latest))

    def _codes_for(self, accounts: Series) -> np.ndarray:
        """Map account identifiers to dense integer positions, extending as needed.

        The Python loop runs once per *newly seen account*, not once per row, so
        its total cost over a run is bounded by the account population.
        """
        text = accounts.astype(str)
        mapped = text.map(self._index)
        unseen = mapped.isna()
        if bool(unseen.any()):
            for value in pd.unique(text[unseen]):
                if value not in self._index:
                    self._index[value] = len(self._index)
            self._ensure_capacity(len(self._index))
            mapped = text.map(self._index)
        return mapped.to_numpy(dtype=np.int64)

    def _ensure_capacity(self, required: int) -> None:
        """Grow the state arrays geometrically so appends stay amortised O(1)."""
        if required <= self._capacity:
            return
        capacity = self._capacity
        while capacity < required:
            capacity *= 2

        self._sent.grow(capacity)
        self._received.grow(capacity)
        self._capacity = capacity

    def export_profiles(self) -> dict[str, Any]:
        """Export current per-account historical state into AccountProfile dictionary."""
        try:
            from infrastructure.data.features.account_history_store import AccountProfile
        except ModuleNotFoundError:
            from backend.infrastructure.data.features.account_history_store import AccountProfile

        profiles: dict[str, AccountProfile] = {}
        for acc_id, code in self._index.items():
            sent_last = float(self._sent.last[code]) if not np.isnan(self._sent.last[code]) else None
            rec_last = float(self._received.last[code]) if not np.isnan(self._received.last[code]) else None
            profiles[acc_id] = AccountProfile(
                account_id=acc_id,
                sent_count=float(self._sent.count[code]),
                sent_total=float(self._sent.total[code]),
                sent_last_epoch=sent_last,
                received_count=float(self._received.count[code]),
                received_total=float(self._received.total[code]),
                received_last_epoch=rec_last,
            )
        return profiles

    def import_profiles(self, profiles: dict[str, Any]) -> None:
        """Seed builder history state with prior AccountProfile records."""
        if not profiles:
            return
        required = len(profiles)
        self._ensure_capacity(required)
        for acc_id, prof in profiles.items():
            if acc_id not in self._index:
                code = len(self._index)
                self._index[acc_id] = code
            else:
                code = self._index[acc_id]

            self._sent.count[code] = getattr(prof, "sent_count", 0.0)
            self._sent.total[code] = getattr(prof, "sent_total", 0.0)
            sent_last = getattr(prof, "sent_last_epoch", None)
            self._sent.last[code] = sent_last if sent_last is not None else np.nan

            self._received.count[code] = getattr(prof, "received_count", 0.0)
            self._received.total[code] = getattr(prof, "received_total", 0.0)
            rec_last = getattr(prof, "received_last_epoch", None)
            self._received.last[code] = rec_last if rec_last is not None else np.nan

    def __repr__(self) -> str:
        """Show the size of the state, which is what bounds memory."""
        return f"AccountHistoryBuilder({self.distinct_accounts} account(s) tracked)"


def _grown(values: np.ndarray, capacity: int, fill: float) -> np.ndarray:
    """Return a larger array holding ``values``, padded with ``fill``."""
    larger = np.full(capacity, fill, dtype=np.float64)
    larger[: values.size] = values
    return larger
