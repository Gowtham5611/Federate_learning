"""IBM AML HI-Small transaction schema — VERIFIED against the real file.

Every entry below was **measured** from the real ``HI-Small_Trans.csv`` on
2026-09-02 by ``backend/scripts/verify_dataset.py``, whose report is committed at
``reports/dataset_verification_HI-Small.md``. Nothing here is remembered from a
dataset description or inferred from a column name; the module previously
refused to define a schema for exactly that reason, and that refusal is now
resolved by evidence rather than by assumption.

Source of every fact:

* file size 475,664,283 bytes; 5,078,345 data rows; 11 columns; UTF-8, no BOM
* zero nulls in all 11 columns, over a full chunked pass
* closed vocabularies for the three low-cardinality text columns, taken from
  complete value counts (not samples)
* ``Is Laundering`` observed with exactly two values, 0 and 1

**The raw header contains ``Account`` twice.** The file's header line is::

    Timestamp,From Bank,Account,To Bank,Account,Amount Received,...

pandas de-duplicates the second occurrence to ``Account.1`` when parsing, so a
frame produced by :class:`~infrastructure.data.loaders.csv_chunk_loader.CsvChunkLoader`
carries ``Account`` (the sending account) and ``Account.1`` (the receiving
account). This schema describes the **parsed frame**, which is what every
consumer actually sees. Renaming to meaningful names is preprocessing, not
ingestion, and happens in the preprocessing layer.
"""

from __future__ import annotations

from domain.enums.column_data_type import ColumnDataType
from domain.enums.unexpected_column_policy import UnexpectedColumnPolicy
from domain.value_objects.column_spec import ColumnSpec
from domain.value_objects.table_schema import TableSchema

IBM_AML_TRANSACTION_SCHEMA_NAME = "ibm_aml_hi_small_transactions"
"""Name the verified schema carries."""

# --- Raw column names, exactly as the parsed frame presents them --------------

COL_TIMESTAMP = "Timestamp"
COL_FROM_BANK = "From Bank"
COL_FROM_ACCOUNT = "Account"
COL_TO_BANK = "To Bank"
COL_TO_ACCOUNT = "Account.1"
COL_AMOUNT_RECEIVED = "Amount Received"
COL_RECEIVING_CURRENCY = "Receiving Currency"
COL_AMOUNT_PAID = "Amount Paid"
COL_PAYMENT_CURRENCY = "Payment Currency"
COL_PAYMENT_FORMAT = "Payment Format"
COL_IS_LAUNDERING = "Is Laundering"

RAW_COLUMN_ORDER: tuple[str, ...] = (
    COL_TIMESTAMP,
    COL_FROM_BANK,
    COL_FROM_ACCOUNT,
    COL_TO_BANK,
    COL_TO_ACCOUNT,
    COL_AMOUNT_RECEIVED,
    COL_RECEIVING_CURRENCY,
    COL_AMOUNT_PAID,
    COL_PAYMENT_CURRENCY,
    COL_PAYMENT_FORMAT,
    COL_IS_LAUNDERING,
)
"""The 11 columns in the order the parsed frame presents them."""

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
"""Verified format of every ``Timestamp`` value (16 characters, minute resolution)."""

# --- Closed vocabularies, from complete value counts over all 5,078,345 rows ---

CURRENCIES: tuple[str, ...] = (
    "Australian Dollar",
    "Bitcoin",
    "Brazil Real",
    "Canadian Dollar",
    "Euro",
    "Mexican Peso",
    "Ruble",
    "Rupee",
    "Saudi Riyal",
    "Shekel",
    "Swiss Franc",
    "UK Pound",
    "US Dollar",
    "Yen",
    "Yuan",
)
"""The 15 currency values observed in both currency columns. Sorted for
determinism; the file's own ordering carries no meaning."""

PAYMENT_FORMATS: tuple[str, ...] = (
    "ACH",
    "Bitcoin",
    "Cash",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
)
"""The 7 payment formats observed. Sorted for determinism."""

TARGET_COLUMN = COL_IS_LAUNDERING
"""The verified label column: two values, 0 and 1, over the whole file."""


def build_ibm_aml_transaction_schema() -> TableSchema:
    """Return the verified schema for the HI-Small transaction file.

    Bounds are stated only where a value outside them would be definitionally
    invalid rather than merely unobserved. Bank identifiers and amounts get a
    lower bound of zero (a negative bank id or a negative transferred amount is
    not a thing) but **no upper bound**: the observed maxima describe this file,
    and asserting them as limits would reject a legitimate larger transaction in
    another variant.

    ``UnexpectedColumnPolicy.REJECT`` is chosen because the file's structure was
    measured exactly and this project consumes one fixed variant: a twelfth
    column means the file is not the one that was verified, which is worth
    failing over.

    Returns:
        The schema describing the parsed frame's 11 columns.
    """
    return TableSchema(
        name=IBM_AML_TRANSACTION_SCHEMA_NAME,
        unexpected_column_policy=UnexpectedColumnPolicy.REJECT,
        columns=(
            ColumnSpec(
                name=COL_TIMESTAMP,
                data_type=ColumnDataType.DATETIME,
                description=(
                    "Transaction time to the minute, formatted "
                    "'YYYY/MM/DD HH:MM'. Observed range 2022/09/01 00:00 to "
                    "2022/09/18 16:18 across 15,018 distinct values. Rows are "
                    "NOT stored in timestamp order."
                ),
            ),
            ColumnSpec(
                name=COL_FROM_BANK,
                data_type=ColumnDataType.INTEGER,
                minimum=0,
                description="Sending bank identifier. Observed 1 to 356,303.",
            ),
            ColumnSpec(
                name=COL_FROM_ACCOUNT,
                data_type=ColumnDataType.STRING,
                description=(
                    "Sending account identifier; 9 hexadecimal characters. "
                    "First of the two columns the raw header names 'Account'."
                ),
            ),
            ColumnSpec(
                name=COL_TO_BANK,
                data_type=ColumnDataType.INTEGER,
                minimum=0,
                description="Receiving bank identifier. Observed 1 to 356,294.",
            ),
            ColumnSpec(
                name=COL_TO_ACCOUNT,
                data_type=ColumnDataType.STRING,
                description=(
                    "Receiving account identifier; 9 hexadecimal characters. "
                    "Second column the raw header names 'Account'; pandas "
                    "de-duplicates it to 'Account.1' when parsing."
                ),
            ),
            ColumnSpec(
                name=COL_AMOUNT_RECEIVED,
                data_type=ColumnDataType.FLOAT,
                minimum=0,
                description=(
                    "Amount credited, denominated in 'Receiving Currency'. "
                    "Observed 1e-06 to 1.0463e+12; no non-positive value exists."
                ),
            ),
            ColumnSpec(
                name=COL_RECEIVING_CURRENCY,
                data_type=ColumnDataType.STRING,
                allowed_values=CURRENCIES,
                description="Currency of 'Amount Received'. 15 observed values.",
            ),
            ColumnSpec(
                name=COL_AMOUNT_PAID,
                data_type=ColumnDataType.FLOAT,
                minimum=0,
                description=(
                    "Amount debited, denominated in 'Payment Currency'. "
                    "Observed 1e-06 to 1.0463e+12; no non-positive value exists."
                ),
            ),
            ColumnSpec(
                name=COL_PAYMENT_CURRENCY,
                data_type=ColumnDataType.STRING,
                allowed_values=CURRENCIES,
                description="Currency of 'Amount Paid'. 15 observed values.",
            ),
            ColumnSpec(
                name=COL_PAYMENT_FORMAT,
                data_type=ColumnDataType.STRING,
                allowed_values=PAYMENT_FORMATS,
                description="Payment instrument. 7 observed values.",
            ),
            ColumnSpec(
                name=COL_IS_LAUNDERING,
                data_type=ColumnDataType.INTEGER,
                minimum=0,
                maximum=1,
                description=(
                    "Target. Verified binary: 0 appears 5,073,168 times "
                    "(99.8981%) and 1 appears 5,177 times (0.1019%). No other "
                    "value occurs. 1 marks a transaction the dataset labels as "
                    "laundering."
                ),
            ),
        ),
    )
