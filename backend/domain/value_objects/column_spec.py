"""Immutable specification of one expected column in a tabular dataset."""

from __future__ import annotations

from dataclasses import dataclass

from domain.enums.column_data_type import ColumnDataType
from domain.exceptions.data import SchemaDefinitionError


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """What the project expects of a single column.

    A specification is a *claim about the data*, so every field it carries must
    be knowable from the real file. Nothing here may be filled in speculatively:
    an ``allowed_values`` set guessed from a dataset description would cause
    valid rows to be rejected as invalid, which is worse than declaring no
    constraint at all.

    ``allowed_values`` is restricted to :attr:`ColumnDataType.STRING` columns on
    purpose. Comparing a parsed numeric column against string literals depends on
    the parser's formatting (``1`` vs ``1.0``), so numeric restrictions are
    expressed as :attr:`minimum`/:attr:`maximum` bounds instead — a 0/1 label
    column is ``INTEGER`` with ``minimum=0, maximum=1``.

    Attributes:
        name: Column header exactly as it appears in the file, including spaces
            and capitalisation. Not normalised: renaming is preprocessing.
        data_type: Logical type the values must conform to.
        required: Whether the column must be present. A missing required column
            is an error; a missing optional column is not.
        nullable: Whether empty values are permitted. Independent of
            ``required``: an optional column that is present may still be
            forbidden from containing nulls.
        allowed_values: Closed vocabulary of permitted values. Empty means
            unconstrained. Only valid for ``STRING`` columns.
        minimum: Inclusive lower bound. Only valid for numeric columns.
        maximum: Inclusive upper bound. Only valid for numeric columns.
        description: What the column means, for documentation and error context.
    """

    name: str
    data_type: ColumnDataType
    required: bool = True
    nullable: bool = False
    allowed_values: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Reject specifications that cannot be enforced coherently.

        Raises:
            SchemaDefinitionError: If the name is blank, a constraint is applied
                to a type that cannot support it, ``allowed_values`` contains
                blanks or duplicates, or the bounds are inverted.
        """
        if not self.name.strip():
            raise SchemaDefinitionError("a column name must not be empty")

        self._validate_allowed_values()
        self._validate_bounds()

    def _validate_allowed_values(self) -> None:
        """Check the closed vocabulary is usable and applied to a text column."""
        if not self.allowed_values:
            return
        if self.data_type is not ColumnDataType.STRING:
            raise SchemaDefinitionError(
                f"column '{self.name}' declares allowed_values but has type "
                f"'{self.data_type.value}'; a closed vocabulary is only supported "
                f"for '{ColumnDataType.STRING.value}' columns, use minimum/maximum "
                f"for numeric restrictions"
            )
        if any(not value.strip() for value in self.allowed_values):
            raise SchemaDefinitionError(
                f"column '{self.name}' has a blank entry in allowed_values; "
                f"use nullable=True to permit empty values"
            )
        if len(set(self.allowed_values)) != len(self.allowed_values):
            raise SchemaDefinitionError(
                f"column '{self.name}' has duplicate entries in allowed_values"
            )

    def _validate_bounds(self) -> None:
        """Check numeric bounds are applied to a numeric column and ordered."""
        has_bound = self.minimum is not None or self.maximum is not None
        if has_bound and not self.data_type.is_numeric:
            raise SchemaDefinitionError(
                f"column '{self.name}' declares a minimum/maximum but has type "
                f"'{self.data_type.value}'; bounds require a numeric type"
            )
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise SchemaDefinitionError(
                f"column '{self.name}' has minimum {self.minimum} greater than "
                f"maximum {self.maximum}"
            )
