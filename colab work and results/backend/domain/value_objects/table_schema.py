"""Immutable specification of an expected tabular structure."""

from __future__ import annotations

from dataclasses import dataclass

from domain.enums.unexpected_column_policy import UnexpectedColumnPolicy
from domain.exceptions.data import SchemaDefinitionError
from domain.value_objects.column_spec import ColumnSpec


@dataclass(frozen=True, slots=True)
class TableSchema:
    """The set of columns a data file is expected to contain.

    A schema is pure data: it describes structure and says nothing about how the
    check is performed. That keeps the definition independent of pandas, so a
    schema can be asserted on, serialised or documented without a DataFrame
    anywhere in sight, and the validation engine can change without touching
    any schema.

    ``unexpected_column_policy`` has no default. The two possible behaviours —
    reject an undeclared column, or accept it as a designed extension — are both
    consequential, and the failure mode of guessing wrong is silent. A schema
    must therefore state which it means.

    Attributes:
        name: Identifies the schema in error messages and logs.
        columns: Declared columns, in the order they are expected to appear.
            Order is documentation only; validation is by name, because a CSV
            with correct headers in a different order is still readable.
        unexpected_column_policy: How to treat a column not declared here.
    """

    name: str
    columns: tuple[ColumnSpec, ...]
    unexpected_column_policy: UnexpectedColumnPolicy

    def __post_init__(self) -> None:
        """Reject schemas that cannot describe a real table.

        Raises:
            SchemaDefinitionError: If the name is blank, no columns are
                declared, or a column name is declared twice.
        """
        if not self.name.strip():
            raise SchemaDefinitionError("a schema name must not be empty")
        if not self.columns:
            raise SchemaDefinitionError(
                "a schema must declare at least one column, otherwise it "
                "asserts nothing about the data",
                self.name,
            )

        seen: set[str] = set()
        duplicates: list[str] = []
        for column in self.columns:
            if column.name in seen:
                duplicates.append(column.name)
            seen.add(column.name)
        if duplicates:
            raise SchemaDefinitionError(
                f"duplicate column definitions: {', '.join(sorted(set(duplicates)))}",
                self.name,
            )

    @property
    def column_names(self) -> tuple[str, ...]:
        """Declared column names, in declaration order."""
        return tuple(column.name for column in self.columns)

    @property
    def required_columns(self) -> tuple[ColumnSpec, ...]:
        """Columns whose absence is a validation error."""
        return tuple(column for column in self.columns if column.required)
