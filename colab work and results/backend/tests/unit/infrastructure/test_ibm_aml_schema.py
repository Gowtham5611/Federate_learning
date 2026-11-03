"""Tests for the IBM AML transaction schema module.

The schema was previously **undefined**, and the tests here held that state in
place: they asserted that requesting it failed loudly, and that the module
invented no columns a later reader could mistake for fact. That blockage is now
resolved — the real ``HI-Small_Trans.csv`` was read in a bounded chunked pass and
``reports/dataset_verification_HI-Small.md`` records what it contains.

These tests were rewritten alongside the definition, as the previous version
said they would have to be. Their job has changed accordingly: they no longer
guard a refusal, they pin the *measured* facts, so that a later edit which
quietly substitutes a plausible-looking column name or a remembered class
balance fails here rather than propagating into the processed data.

Every number asserted below appears in the committed verification report. None
of it is checked against the real file at test time — that would need the
475 MB dataset, which the unit suite never touches.
"""

from __future__ import annotations

import inspect

import pytest

from domain.enums.column_data_type import ColumnDataType
from domain.enums.unexpected_column_policy import UnexpectedColumnPolicy
from domain.value_objects.table_schema import TableSchema
from infrastructure.data.schemas import ibm_aml_schema
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
    CURRENCIES,
    IBM_AML_TRANSACTION_SCHEMA_NAME,
    PAYMENT_FORMATS,
    RAW_COLUMN_ORDER,
    TARGET_COLUMN,
    TIMESTAMP_FORMAT,
    build_ibm_aml_transaction_schema,
)

pytestmark = pytest.mark.unit


# --- The schema exists and describes the file that was verified ----------------


def test_building_the_schema_returns_a_table_schema() -> None:
    """The refusal is gone: the columns were measured, so they can be stated."""
    assert isinstance(build_ibm_aml_transaction_schema(), TableSchema)


def test_the_schema_keeps_the_name_that_was_reserved_for_it() -> None:
    """Named before it was defined, so the definition and its docs agree."""
    assert IBM_AML_TRANSACTION_SCHEMA_NAME == "ibm_aml_hi_small_transactions"
    assert build_ibm_aml_transaction_schema().name == IBM_AML_TRANSACTION_SCHEMA_NAME


def test_the_schema_declares_exactly_the_eleven_verified_columns() -> None:
    """Eleven was counted in the real header, not assumed from a description."""
    schema = build_ibm_aml_transaction_schema()

    assert len(schema.columns) == 11
    assert tuple(column.name for column in schema.columns) == RAW_COLUMN_ORDER


def test_the_two_account_columns_carry_the_names_pandas_actually_produces() -> None:
    """The raw header says 'Account' twice; pandas renames the second.

    A schema written against the *header* rather than the *parsed frame* would
    look right and match nothing, because no consumer ever sees two columns
    called ``Account``.
    """
    assert COL_FROM_ACCOUNT == "Account"
    assert COL_TO_ACCOUNT == "Account.1"


def test_the_column_order_matches_the_verified_header_order() -> None:
    assert RAW_COLUMN_ORDER == (
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


def test_the_timestamp_format_is_the_one_observed_in_the_file() -> None:
    """Minute resolution, slash-separated. Parsing depends on getting this right."""
    assert TIMESTAMP_FORMAT == "%Y/%m/%d %H:%M"


# --- Vocabularies come from complete counts, not from samples ------------------


def test_the_currency_vocabulary_holds_the_fifteen_observed_values() -> None:
    assert len(CURRENCIES) == 15
    assert "US Dollar" in CURRENCIES
    assert "Bitcoin" in CURRENCIES


def test_the_payment_format_vocabulary_holds_the_seven_observed_values() -> None:
    assert len(PAYMENT_FORMATS) == 7
    assert set(PAYMENT_FORMATS) == {
        "ACH",
        "Bitcoin",
        "Cash",
        "Cheque",
        "Credit Card",
        "Reinvestment",
        "Wire",
    }


def test_both_vocabularies_are_sorted_so_encodings_are_reproducible() -> None:
    """Downstream codes are assigned in sorted order; unsorted input would drift."""
    assert list(CURRENCIES) == sorted(CURRENCIES)
    assert list(PAYMENT_FORMATS) == sorted(PAYMENT_FORMATS)


def test_both_currency_columns_share_the_one_vocabulary() -> None:
    schema = build_ibm_aml_transaction_schema()
    columns = {column.name: column for column in schema.columns}

    assert columns[COL_RECEIVING_CURRENCY].allowed_values == CURRENCIES
    assert columns[COL_PAYMENT_CURRENCY].allowed_values == CURRENCIES


def test_the_payment_format_column_is_the_only_other_closed_vocabulary() -> None:
    """Account and bank identifiers are open sets.

    Constraining them would reject a valid file.
    """
    schema = build_ibm_aml_transaction_schema()
    constrained = {
        column.name for column in schema.columns if column.allowed_values
    }

    assert constrained == {
        COL_RECEIVING_CURRENCY,
        COL_PAYMENT_CURRENCY,
        COL_PAYMENT_FORMAT,
    }


# --- The target ----------------------------------------------------------------


def test_the_target_is_the_column_the_file_actually_labels_with() -> None:
    assert TARGET_COLUMN == "Is Laundering"
    assert TARGET_COLUMN in RAW_COLUMN_ORDER


def test_the_target_is_bounded_to_the_two_values_that_occur() -> None:
    """0 and 1 were counted over all 5,078,345 rows; nothing else appears."""
    schema = build_ibm_aml_transaction_schema()
    target = next(column for column in schema.columns if column.name == TARGET_COLUMN)

    assert target.data_type is ColumnDataType.INTEGER
    assert target.minimum == 0
    assert target.maximum == 1


# --- Bounds are stated only where they are definitional ------------------------


def test_amounts_and_banks_are_bounded_below_but_not_above() -> None:
    """An upper bound would be this file's maximum mistaken for a rule.

    A negative transferred amount is not a magnitude, so zero is a real floor.
    The largest amount observed here is a fact about these 5,078,345 rows, and
    asserting it as a limit would reject a legitimately larger transaction.
    """
    schema = build_ibm_aml_transaction_schema()
    columns = {column.name: column for column in schema.columns}

    for name in (COL_FROM_BANK, COL_TO_BANK, COL_AMOUNT_PAID, COL_AMOUNT_RECEIVED):
        assert columns[name].minimum == 0, name
        assert columns[name].maximum is None, name


def test_account_identifiers_are_text_not_numbers() -> None:
    """They are nine hexadecimal characters; reading them as numbers loses rows."""
    schema = build_ibm_aml_transaction_schema()
    columns = {column.name: column for column in schema.columns}

    assert columns[COL_FROM_ACCOUNT].data_type is ColumnDataType.STRING
    assert columns[COL_TO_ACCOUNT].data_type is ColumnDataType.STRING


def test_every_column_is_required_and_non_nullable() -> None:
    """A full pass measured zero nulls in all eleven columns."""
    schema = build_ibm_aml_transaction_schema()

    assert all(column.required for column in schema.columns)
    assert not any(column.nullable for column in schema.columns)


def test_an_unexpected_column_is_rejected_rather_than_ignored() -> None:
    """A twelfth column means this is not the file that was verified."""
    schema = build_ibm_aml_transaction_schema()

    assert schema.unexpected_column_policy is UnexpectedColumnPolicy.REJECT


# --- The module stays auditable ------------------------------------------------


def test_every_column_carries_a_description_of_what_was_measured() -> None:
    """The description is where a reader checks a value against the report."""
    schema = build_ibm_aml_transaction_schema()

    assert all(column.description for column in schema.columns)


def test_the_module_cites_the_verification_report_as_its_source() -> None:
    """A schema nobody can trace back to evidence is indistinguishable from a guess."""
    doc = ibm_aml_schema.__doc__ or ""

    assert "reports/dataset_verification_HI-Small.md" in doc
    assert "HI-Small_Trans.csv" in doc
    assert "5,078,345" in doc


def test_the_module_records_why_the_duplicated_account_header_matters() -> None:
    doc = ibm_aml_schema.__doc__ or ""

    assert "Account.1" in doc


def test_the_builder_explains_why_it_states_no_upper_bounds() -> None:
    """The reasoning belongs next to the decision, not only in a report."""
    doc = inspect.getdoc(build_ibm_aml_transaction_schema) or ""

    assert "upper bound" in doc
    assert "REJECT" in doc


def test_building_the_schema_twice_produces_equal_schemas() -> None:
    """No hidden state, so two runs of the pipeline validate identically."""
    assert build_ibm_aml_transaction_schema() == build_ibm_aml_transaction_schema()
