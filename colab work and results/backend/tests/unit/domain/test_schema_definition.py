"""Tests for the schema value objects.

These cover the schema *definition*: what a schema will and will not accept as a
coherent description. Whether data conforms to one is tested separately, in
``tests/unit/infrastructure/test_schema_validator.py``.
"""

from __future__ import annotations

import pytest

from domain.enums.column_data_type import ColumnDataType
from domain.enums.unexpected_column_policy import UnexpectedColumnPolicy
from domain.exceptions.data import SchemaDefinitionError
from domain.value_objects.column_spec import ColumnSpec
from domain.value_objects.table_schema import TableSchema
from tests.fixtures.tabular_fixtures import SYNTHETIC_COLUMNS, make_synthetic_schema

pytestmark = pytest.mark.unit


# --- A coherent schema is accepted (required test 16, definition half) --------


def test_a_valid_schema_is_accepted_and_reports_its_columns() -> None:
    schema = make_synthetic_schema()

    assert schema.column_names == SYNTHETIC_COLUMNS
    assert schema.unexpected_column_policy is UnexpectedColumnPolicy.REJECT


def test_required_columns_excludes_optional_ones() -> None:
    schema = make_synthetic_schema()

    required = tuple(spec.name for spec in schema.required_columns)

    assert "record_id" in required
    assert "receiver_ref" not in required, "the optional column must not be required"


def test_column_order_is_preserved_as_declared() -> None:
    """Order is documentation, but it must not be silently reshuffled."""
    assert make_synthetic_schema().column_names == SYNTHETIC_COLUMNS


# --- Column specifications reject what they cannot enforce --------------------


def test_blank_column_name_is_rejected() -> None:
    with pytest.raises(SchemaDefinitionError, match="must not be empty"):
        ColumnSpec(name="   ", data_type=ColumnDataType.STRING)


@pytest.mark.parametrize(
    "data_type",
    [ColumnDataType.INTEGER, ColumnDataType.FLOAT, ColumnDataType.DATETIME],
)
def test_allowed_values_on_a_non_string_column_is_rejected(
    data_type: ColumnDataType,
) -> None:
    """A closed vocabulary compared against parsed numbers is a formatting trap."""
    with pytest.raises(SchemaDefinitionError, match="closed vocabulary"):
        ColumnSpec(name="c", data_type=data_type, allowed_values=("1", "2"))


def test_blank_entry_in_allowed_values_is_rejected() -> None:
    with pytest.raises(SchemaDefinitionError, match="blank entry"):
        ColumnSpec(name="c", data_type=ColumnDataType.STRING, allowed_values=("A", " "))


def test_duplicate_allowed_values_are_rejected() -> None:
    with pytest.raises(SchemaDefinitionError, match="duplicate entries"):
        ColumnSpec(name="c", data_type=ColumnDataType.STRING, allowed_values=("A", "A"))


@pytest.mark.parametrize(
    "data_type", [ColumnDataType.STRING, ColumnDataType.BOOLEAN, ColumnDataType.DATETIME]
)
def test_bounds_on_a_non_numeric_column_are_rejected(data_type: ColumnDataType) -> None:
    with pytest.raises(SchemaDefinitionError, match="bounds require a numeric type"):
        ColumnSpec(name="c", data_type=data_type, minimum=0)


def test_inverted_bounds_are_rejected() -> None:
    with pytest.raises(SchemaDefinitionError, match="greater than"):
        ColumnSpec(name="c", data_type=ColumnDataType.FLOAT, minimum=10, maximum=1)


def test_equal_bounds_are_accepted() -> None:
    """A single permitted value is a legitimate, if narrow, constraint."""
    spec = ColumnSpec(name="c", data_type=ColumnDataType.INTEGER, minimum=1, maximum=1)

    assert (spec.minimum, spec.maximum) == (1, 1)


def test_a_column_specification_is_immutable() -> None:
    spec = ColumnSpec(name="c", data_type=ColumnDataType.STRING)

    with pytest.raises((AttributeError, TypeError)):
        spec.name = "other"  # type: ignore[misc]


# --- Table schemas reject what cannot describe a table -----------------------


def test_blank_schema_name_is_rejected() -> None:
    with pytest.raises(SchemaDefinitionError, match="must not be empty"):
        TableSchema(
            name="  ",
            columns=(ColumnSpec(name="c", data_type=ColumnDataType.STRING),),
            unexpected_column_policy=UnexpectedColumnPolicy.REJECT,
        )


def test_a_schema_with_no_columns_is_rejected() -> None:
    with pytest.raises(SchemaDefinitionError, match="at least one column"):
        TableSchema(
            name="empty",
            columns=(),
            unexpected_column_policy=UnexpectedColumnPolicy.REJECT,
        )


def test_duplicate_column_definitions_are_rejected() -> None:
    """Two specs for one name make validation's outcome depend on iteration order."""
    with pytest.raises(SchemaDefinitionError, match="duplicate column"):
        TableSchema(
            name="dupes",
            columns=(
                ColumnSpec(name="c", data_type=ColumnDataType.STRING),
                ColumnSpec(name="c", data_type=ColumnDataType.INTEGER),
            ),
            unexpected_column_policy=UnexpectedColumnPolicy.REJECT,
        )


def test_the_unexpected_column_policy_must_be_stated_explicitly() -> None:
    """There is no default: both behaviours are consequential."""
    with pytest.raises(TypeError):
        TableSchema(  # type: ignore[call-arg]
            name="no-policy",
            columns=(ColumnSpec(name="c", data_type=ColumnDataType.STRING),),
        )


def test_schemas_with_equal_definitions_compare_equal() -> None:
    """Value semantics: a schema is data, so two identical ones are the same."""
    assert make_synthetic_schema() == make_synthetic_schema()
    assert make_synthetic_schema() != make_synthetic_schema(
        policy=UnexpectedColumnPolicy.ALLOW
    )
