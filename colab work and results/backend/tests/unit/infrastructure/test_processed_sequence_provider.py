"""Tests for :mod:`infrastructure.data.sequences.processed_sequence_provider`.

The provider is the boundary between Phase 3's files and Phase 4's tensors, so the
tests are about what crosses that boundary and what does not:

- the feature matrix contains the recorded feature columns and nothing else — no
  timestamp, no account identifier, no label;
- the nominal codes are expanded using the *recorded* vocabulary, so every split
  comes out the same width even when a category is absent from one of them;
- an out-of-order split is refused rather than silently producing windows in which
  a later transaction precedes an earlier one;
- the three architectures are handed byte-identical tensors.

Every test reads a synthetic tree under ``tmp_path``. The real 530 MB processed
tree is never touched here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from domain.exceptions.training import (
    EmptySequenceDatasetError,
    ProcessedMetadataError,
    UnsortedSplitError,
)
from infrastructure.data.sequences.processed_sequence_provider import (
    BOUNDARY_POLICY,
    ProcessedSequenceProvider,
)
from tests.fixtures.sequence_fixtures import (
    CONTINUOUS_COLUMNS,
    EXPANDED_DIMENSION,
    FEATURE_COLUMNS,
    NOMINAL_COLUMN,
    PREPROCESSING_VERSION,
    SENDER_COLUMN,
    SOURCE_DIMENSION,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    UNKNOWN_CODE,
    make_rows,
    write_processed_tree,
    write_split,
)

pytestmark = pytest.mark.unit


def make_config(**overrides: object) -> TrainingConfig:
    """A configuration small enough for a 60-row fixture split."""
    values: dict[str, object] = {
        "architecture": ModelArchitecture.GRU,
        "sequence_length": 4,
        "batch_size": 8,
        "epochs": 1,
        "hidden_size": 4,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A complete synthetic processed tree."""
    return write_processed_tree(tmp_path / "processed")


@pytest.fixture
def provider(tree: Path) -> ProcessedSequenceProvider:
    """A provider over the synthetic tree."""
    return ProcessedSequenceProvider(tree)


# --- What reaches the feature matrix -----------------------------------------


def test_feature_matrix_is_the_recorded_width(
    provider: ProcessedSequenceProvider,
) -> None:
    dataset = provider.load(SplitName.TRAIN, make_config())

    assert dataset.feature_dimension == EXPANDED_DIMENSION


def test_expansion_can_be_switched_off(provider: ProcessedSequenceProvider) -> None:
    dataset = provider.load(
        SplitName.TRAIN, make_config(one_hot_nominal_codes=False)
    )

    assert dataset.feature_dimension == SOURCE_DIMENSION


def test_continuous_values_are_carried_through_unchanged(
    provider: ProcessedSequenceProvider,
) -> None:
    dataset = provider.load(SplitName.TRAIN, make_config(sequence_length=1))
    batch = next(iter(dataset.batches(4, shuffle=False)))
    rows = make_rows(rows=120)

    # sequence_length 1 means each window is exactly its own target row, so
    # position 0 of the window is that row's feature vector.
    for position in range(4):
        assert batch.features[position, 0, 0] == pytest.approx(
            rows[position]["amount_log"]
        )
        assert batch.features[position, 0, 1] == pytest.approx(
            float(rows[position]["same_bank"])
        )


def test_nominal_code_becomes_a_one_hot_block(
    provider: ProcessedSequenceProvider,
) -> None:
    dataset = provider.load(SplitName.TRAIN, make_config(sequence_length=1))
    batch = next(iter(dataset.batches(6, shuffle=False)))
    rows = make_rows(rows=120)

    for position in range(6):
        block = batch.features[position, 0, len(CONTINUOUS_COLUMNS) :]
        expected = np.zeros(3, dtype=np.float32)
        expected[int(rows[position][NOMINAL_COLUMN])] = 1.0
        assert block.tolist() == expected.tolist()
        # Exactly one indicator is set: a code cannot be two categories at once.
        assert block.sum() == pytest.approx(1.0)


def test_out_of_vocabulary_code_expands_to_all_zeros(tmp_path: Path) -> None:
    # The unknown code is what preprocessing writes for a category absent from the
    # fitted vocabulary. "None of the known categories" is an all-zero block, not a
    # category of its own.
    rows = make_rows(rows=10)
    rows[3][NOMINAL_COLUMN] = UNKNOWN_CODE
    root = write_processed_tree(tmp_path / "processed", splits={SplitName.TRAIN: rows})
    dataset = ProcessedSequenceProvider(root).load(
        SplitName.TRAIN, make_config(sequence_length=1)
    )

    batch = next(iter(dataset.batches(10, shuffle=False)))
    block = batch.features[3, 0, len(CONTINUOUS_COLUMNS) :]
    assert block.tolist() == [0.0, 0.0, 0.0]
    # The width is unchanged: an unknown code does not widen the tensor.
    assert dataset.feature_dimension == EXPANDED_DIMENSION


def test_width_comes_from_the_vocabulary_not_from_the_values_present(
    tmp_path: Path,
) -> None:
    # A split in which only one category ever occurs must still produce the full
    # three-wide block, or train and test would disagree on the tensor shape.
    rows = make_rows(rows=12, codes=1)
    root = write_processed_tree(tmp_path / "processed", splits={SplitName.TRAIN: rows})
    dataset = ProcessedSequenceProvider(root).load(SplitName.TRAIN, make_config())

    assert dataset.feature_dimension == EXPANDED_DIMENSION


def test_all_splits_come_out_the_same_width(
    provider: ProcessedSequenceProvider,
) -> None:
    config = make_config()
    widths = {
        split: provider.load(split, config).feature_dimension for split in SplitName
    }

    assert len(set(widths.values())) == 1


# --- What must not reach the feature matrix ----------------------------------


def test_identifier_and_target_columns_are_not_on_the_feature_axis(
    provider: ProcessedSequenceProvider,
) -> None:
    columns = provider.metadata.model_columns(one_hot_nominal_codes=True)

    assert TIMESTAMP_COLUMN not in columns
    assert SENDER_COLUMN not in columns
    assert TARGET_COLUMN not in columns


def test_the_label_is_not_recoverable_from_the_features(tmp_path: Path) -> None:
    # Two trees differing only in the labels must produce identical feature
    # tensors. If the target leaked into the matrix, they would differ.
    clean = make_rows(rows=20, positive_every=0)
    labelled = [dict(row) for row in clean]
    for row in labelled:
        row[TARGET_COLUMN] = 1

    config = make_config()
    tensors = []
    for name, rows in (("a", clean), ("b", labelled)):
        root = write_processed_tree(
            tmp_path / name, splits={SplitName.TRAIN: rows}
        )
        dataset = ProcessedSequenceProvider(root).load(SplitName.TRAIN, config)
        batch = next(iter(dataset.batches(20, shuffle=False)))
        tensors.append(batch.features.copy())

    assert np.array_equal(tensors[0], tensors[1])
    # ...while the labels themselves did change, so the fixture is a real test.
    assert not np.array_equal(clean[0][TARGET_COLUMN], labelled[0][TARGET_COLUMN])


def test_account_identity_survives_only_as_history_shape(tmp_path: Path) -> None:
    # Renaming every account, keeping the interleaving, must not change the
    # tensor: what the provider keeps is the shape of each history, not the name.
    original = make_rows(rows=18)
    renamed = [dict(row) for row in original]
    for row in renamed:
        row[SENDER_COLUMN] = f"ZZZ{row[SENDER_COLUMN]}"

    config = make_config()
    outputs = []
    for name, rows in (("orig", original), ("renamed", renamed)):
        root = write_processed_tree(tmp_path / name, splits={SplitName.TRAIN: rows})
        dataset = ProcessedSequenceProvider(root).load(SplitName.TRAIN, config)
        batch = next(iter(dataset.batches(18, shuffle=False)))
        outputs.append((batch.features.copy(), batch.lengths.copy()))

    assert np.array_equal(outputs[0][0], outputs[1][0])
    assert np.array_equal(outputs[0][1], outputs[1][1])


# --- Order and boundaries ----------------------------------------------------


def test_an_out_of_order_split_is_refused(tmp_path: Path) -> None:
    rows = make_rows(rows=10)
    rows[4], rows[7] = rows[7], rows[4]
    root = write_processed_tree(tmp_path / "processed", splits={SplitName.TRAIN: rows})

    with pytest.raises(UnsortedSplitError) as caught:
        ProcessedSequenceProvider(root).load(SplitName.TRAIN, make_config())

    assert "train" in str(caught.value)


def test_the_first_offending_row_is_named(tmp_path: Path) -> None:
    rows = make_rows(rows=8)
    # Move the last timestamp to row 2, so row 3 is the first that goes backwards.
    rows[2][TIMESTAMP_COLUMN] = rows[7][TIMESTAMP_COLUMN]
    root = write_processed_tree(tmp_path / "processed", splits={SplitName.TRAIN: rows})

    with pytest.raises(UnsortedSplitError) as caught:
        ProcessedSequenceProvider(root).load(SplitName.TRAIN, make_config())

    assert "3" in str(caught.value)


def test_equal_timestamps_are_accepted(tmp_path: Path) -> None:
    # Ties are not disorder: the raw dataset records many transactions in the same
    # minute, and refusing them would refuse the real data.
    rows = make_rows(rows=10, minutes_apart=5)
    rows[5][TIMESTAMP_COLUMN] = rows[4][TIMESTAMP_COLUMN]
    root = write_processed_tree(tmp_path / "processed", splits={SplitName.TRAIN: rows})

    dataset = ProcessedSequenceProvider(root).load(SplitName.TRAIN, make_config())
    assert dataset.sequence_count == 10


def test_sequences_never_span_a_split_boundary(
    provider: ProcessedSequenceProvider,
) -> None:
    config = make_config(sequence_length=6)
    train = provider.load(SplitName.TRAIN, config)
    validation = provider.load(SplitName.VALIDATION, config)

    # Each dataset knows only its own rows, so the largest row a validation window
    # can name is validation's own last row - the boundary is structural.
    assert train.row_count == 120
    assert validation.row_count == 60
    for batch in validation.batches(60, shuffle=False):
        assert int(batch.row_indices.max()) < validation.row_count


def test_boundary_policy_is_stated_in_the_strategy(
    provider: ProcessedSequenceProvider,
) -> None:
    strategy = provider.describe_strategy(make_config())

    assert strategy.boundary_policy == BOUNDARY_POLICY
    assert "train/validation" in strategy.boundary_policy


def test_history_stops_at_the_start_of_the_split(
    provider: ProcessedSequenceProvider,
) -> None:
    # Validation's first rows have no in-split history, so their windows are short
    # rather than borrowed from train.
    dataset = provider.load(SplitName.VALIDATION, make_config(sequence_length=5))
    batch = next(iter(dataset.batches(6, shuffle=False)))

    assert batch.lengths[0] == 1


# --- The same tensor for every architecture ----------------------------------


@pytest.mark.parametrize("architecture", list(ModelArchitecture))
def test_every_architecture_receives_an_identical_tensor(
    provider: ProcessedSequenceProvider, architecture: ModelArchitecture
) -> None:
    reference = provider.load(
        SplitName.TRAIN, make_config(architecture=ModelArchitecture.GRU)
    )
    candidate = provider.load(SplitName.TRAIN, make_config(architecture=architecture))

    reference_batch = next(iter(reference.batches(16, shuffle=False)))
    candidate_batch = next(iter(candidate.batches(16, shuffle=False)))
    assert np.array_equal(reference_batch.features, candidate_batch.features)
    assert np.array_equal(reference_batch.labels, candidate_batch.labels)
    assert np.array_equal(reference_batch.lengths, candidate_batch.lengths)


# --- Counts, provenance and limits -------------------------------------------


def test_row_and_positive_counts_are_measured_as_splits_are_read(
    provider: ProcessedSequenceProvider,
) -> None:
    config = make_config()
    before = provider.describe_features(config)
    assert before.rows_per_split == {}

    provider.load(SplitName.TRAIN, config)
    after = provider.describe_features(config)

    assert after.rows_per_split == {"train": 120}
    # positive_every defaults to 7 over 120 rows: rows 0, 7, ... 119 -> 18.
    assert after.positives_per_split == {"train": 18}


def test_provenance_records_the_directory_and_the_version(
    provider: ProcessedSequenceProvider, tree: Path
) -> None:
    provenance = provider.describe_features(make_config())

    assert Path(provenance.processed_directory) == tree.resolve()
    assert provenance.preprocessing_version == PREPROCESSING_VERSION
    assert provenance.feature_columns == FEATURE_COLUMNS
    assert provenance.target_column == TARGET_COLUMN


def test_provenance_reports_both_the_source_count_and_the_model_width(
    provider: ProcessedSequenceProvider,
) -> None:
    provenance = provider.describe_features(make_config())
    row = provenance.to_dict()

    assert row["source_feature_count"] == SOURCE_DIMENSION
    assert row["model_input_dimension"] == EXPANDED_DIMENSION


def test_row_limit_reads_a_prefix_of_the_split(
    provider: ProcessedSequenceProvider,
) -> None:
    dataset = provider.load(SplitName.TRAIN, make_config(row_limit=25))

    assert dataset.row_count == 25
    assert dataset.sequence_count == 25


def test_strategy_names_the_grouping_column_and_the_configured_shape(
    provider: ProcessedSequenceProvider,
) -> None:
    strategy = provider.describe_strategy(make_config(sequence_length=7, sequence_stride=2))

    assert strategy.grouping_column == SENDER_COLUMN
    assert strategy.sequence_length == 7
    assert strategy.stride == 2
    assert "newest" in strategy.target_assignment


# --- Refusals ----------------------------------------------------------------


def test_a_missing_split_file_is_named(tmp_path: Path) -> None:
    root = write_processed_tree(
        tmp_path / "processed", splits={SplitName.TRAIN: make_rows(rows=10)}
    )

    with pytest.raises(ProcessedMetadataError) as caught:
        ProcessedSequenceProvider(root).load(SplitName.TEST, make_config())

    assert "test" in str(caught.value)


def test_an_empty_split_is_refused(tmp_path: Path) -> None:
    root = write_processed_tree(tmp_path / "processed", splits={SplitName.TRAIN: []})

    with pytest.raises(EmptySequenceDatasetError):
        ProcessedSequenceProvider(root).load(SplitName.TRAIN, make_config())


def test_a_split_missing_a_described_column_is_refused(tmp_path: Path) -> None:
    root = write_processed_tree(
        tmp_path / "processed", splits={SplitName.TRAIN: make_rows(rows=10)}
    )
    # Rewrite the split without the nominal column the metadata describes.
    write_split(
        root,
        SplitName.TRAIN,
        make_rows(rows=10),
        columns=(TIMESTAMP_COLUMN, SENDER_COLUMN, "amount_log", "same_bank", TARGET_COLUMN),
    )

    with pytest.raises(ValueError, match=NOMINAL_COLUMN):
        ProcessedSequenceProvider(root).load(SplitName.TRAIN, make_config())


def test_a_root_without_metadata_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProcessedMetadataError):
        ProcessedSequenceProvider(tmp_path / "nothing_here")
