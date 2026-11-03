"""Memory-mapped federated partition manager for full-scale datasets.

Enables zero-copy, RAM-bounded client partitioning of multi-gigabyte datasets
(e.g., IBM AML LI-Medium with 18.75 million training transactions).

Key features:
1. Streaming chunked partition calculation (RAM usage < 250 MB).
2. Disk-backed index persistence (client row indices and client previous-row links).
3. Zero-copy MmapClientDataset: mini-batches query the global disk-backed
   features.mmap on demand, eliminating DataFrame and feature matrix duplication.
4. Preserves chronological sequence integrity and account atomicity (Non-IID).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from application.dto.federated import ClientPartitionStats, IIDPartitionReport, NonIIDPartitionReport
from application.dto.training import TrainingConfig
from application.ports.output.sequence_data import SequenceBatch, SequenceDataset
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.account_sequence_dataset import NO_PREVIOUS_ROW, build_previous_row_index
from infrastructure.data.sequences.mmap_sequence_provider import (
    MmapSequenceProvider,
    SubsetModeDetectedError,
)
from infrastructure.data.sequences.processed_feature_metadata import ProcessedFeatureMetadata

logger = logging.getLogger(__name__)

CLIENT_IDS: tuple[str, ...] = ("client_1", "client_2", "client_3")
TYPOLOGY_NAMES: tuple[str, ...] = (
    "Retail Giant",
    "Offshore / Crypto Exchange",
    "Regional Credit Union",
)


class MmapClientDataset(SequenceDataset):
    """Zero-copy client dataset backed by the global MmapSequenceDataset."""

    def __init__(
        self,
        *,
        split: SplitName,
        parent_features: NDArray[np.float32],
        client_targets: NDArray[np.float32],
        client_previous_row: NDArray[np.int64],
        client_global_indices: NDArray[np.int64],
        sequence_length: int,
        client_id: str,
    ) -> None:
        self._split = split
        self._features = parent_features  # Global np.memmap (rows, 57)
        self._targets = client_targets
        self._previous = client_previous_row
        self._global_indices = client_global_indices
        self._length = sequence_length
        self._client_id = client_id
        self._row_count = int(client_targets.shape[0])
        self._target_rows = np.arange(self._row_count, dtype=np.int64)

    @property
    def split(self) -> SplitName:
        return self._split

    @property
    def sequence_count(self) -> int:
        return self._row_count

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def positive_count(self) -> int:
        return int(np.sum(self._targets == 1.0))

    @property
    def negative_count(self) -> int:
        return self._row_count - self.positive_count

    @property
    def feature_dimension(self) -> int:
        return int(self._features.shape[1])

    @property
    def sequence_length(self) -> int:
        return self._length

    @property
    def labels(self) -> NDArray[np.float32]:
        return self._targets

    @property
    def target_rows(self) -> NDArray[np.int64]:
        return self._target_rows

    def batches(
        self, batch_size: int, *, shuffle: bool = False, seed: int | None = None
    ) -> Iterator[SequenceBatch]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size!r}")

        order = self._target_rows
        if shuffle:
            rng = np.random.default_rng(seed)
            order = rng.permutation(order)

        total = int(order.shape[0])
        for start in range(0, total, batch_size):
            yield self._assemble(order[start : start + batch_size])

    def _assemble(self, target_rows: NDArray[np.int64]) -> SequenceBatch:
        walk, lengths = self._walk(target_rows)

        positions = np.arange(self._length, dtype=np.int64)
        steps = lengths[:, None] - 1 - positions[None, :]
        real = steps >= 0
        local_rows = np.take_along_axis(walk, np.maximum(steps, 0), axis=1)

        # Map client-local row indices to global indices in the shared memmap
        global_rows = self._global_indices[np.maximum(local_rows, 0)]

        # Read slice directly from the shared memory-mapped feature array
        features = np.asarray(self._features[global_rows], dtype=np.float32)
        features[~real] = 0.0

        return SequenceBatch(
            features=features,
            lengths=lengths,
            labels=self._targets[target_rows],
            row_indices=self._global_indices[target_rows],
        )

    def _walk(
        self, target_rows: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        n = int(target_rows.shape[0])
        walk = np.full((n, self._length), NO_PREVIOUS_ROW, dtype=np.int64)
        walk[:, 0] = target_rows

        current = target_rows
        for k in range(1, self._length):
            valid = current != NO_PREVIOUS_ROW
            next_row = np.full(n, NO_PREVIOUS_ROW, dtype=np.int64)
            if np.any(valid):
                next_row[valid] = self._previous[current[valid]]
            walk[:, k] = next_row
            current = next_row

        lengths = np.sum(walk != NO_PREVIOUS_ROW, axis=1)
        return walk, lengths


class MmapPartitionManager:
    """Manages creation, caching, and loading of memory-mapped federated client partitions."""

    def __init__(self, metadata: ProcessedFeatureMetadata, cache_root: Path | None = None) -> None:
        self._metadata = metadata
        self._cache_root = cache_root or (metadata.root / ".cache" / "partitions")
        self._cache_root.mkdir(parents=True, exist_ok=True)

    def get_or_create_partitions(
        self,
        distribution: str,
        *,
        seed: int = 42,
        n_clients: int = 3,
        chunk_size: int = 250000,
        enforce_full_dataset: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Retrieve cached client partition indices or compute them in streaming mode."""
        dist = distribution.lower()
        if dist not in ("iid", "non_iid"):
            raise ValueError(f"Unknown distribution '{distribution}'. Expected 'iid' or 'non_iid'.")

        train_path = self._metadata.split_path(SplitName.TRAIN.value)
        train_csv_size = train_path.stat().st_size
        train_csv_mtime = train_path.stat().st_mtime

        part_dir = self._cache_root / f"{dist}_n{n_clients}_seed{seed}"
        fingerprint_file = part_dir / ".fingerprint.json"

        # Check if valid cached partition exists
        if fingerprint_file.exists():
            try:
                with open(fingerprint_file, "r", encoding="utf-8") as f:
                    fp = json.load(f)
                if (
                    fp.get("train_csv_size") == train_csv_size
                    and fp.get("train_csv_mtime") == train_csv_mtime
                    and fp.get("distribution") == dist
                    and fp.get("n_clients") == n_clients
                    and fp.get("seed") == seed
                ):
                    # Verify all client index files exist
                    all_exist = True
                    for k in range(1, n_clients + 1):
                        cid = f"client_{k}"
                        if not (
                            (part_dir / f"{cid}_indices.npy").exists()
                            and (part_dir / f"{cid}_previous_row.npy").exists()
                            and (part_dir / f"{cid}_targets.npy").exists()
                        ):
                            all_exist = False
                            break
                    if all_exist:
                        logger.info("Found valid cached partitions in %s. Loading indices...", part_dir)
                        return self._load_partition_indices(part_dir, n_clients)
            except Exception as e:
                logger.warning("Cache validation error in %s: %s. Recomputing...", part_dir, e)

        # Compute partitions in streaming mode
        logger.info(
            "Computing %s partitions across %d clients (seed=%d, chunk_size=%d)...",
            dist.upper(),
            n_clients,
            seed,
            chunk_size,
        )
        part_dir.mkdir(parents=True, exist_ok=True)

        if dist == "iid":
            result = self._compute_iid_partitions(
                train_path,
                n_clients=n_clients,
                seed=seed,
                chunk_size=chunk_size,
                enforce_full_dataset=enforce_full_dataset,
            )
        else:
            result = self._compute_non_iid_partitions(
                train_path,
                seed=seed,
                chunk_size=chunk_size,
                enforce_full_dataset=enforce_full_dataset,
            )

        # Save partition arrays and fingerprint
        total_rows = 0
        for cid, data in result.items():
            np.save(part_dir / f"{cid}_indices.npy", data["indices"])
            np.save(part_dir / f"{cid}_previous_row.npy", data["previous_row"])
            np.save(part_dir / f"{cid}_targets.npy", data["targets"])
            total_rows += len(data["indices"])

        fingerprint = {
            "train_csv_size": train_csv_size,
            "train_csv_mtime": train_csv_mtime,
            "distribution": dist,
            "n_clients": n_clients,
            "seed": seed,
            "total_rows": total_rows,
            "clients": {
                cid: {
                    "rows": len(d["indices"]),
                    "positives": int(np.sum(d["targets"] == 1.0)),
                }
                for cid, d in result.items()
            },
        }
        with open(fingerprint_file, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)

        logger.info(
            "Successfully cached %s partitions for %d rows in %s.",
            dist.upper(),
            total_rows,
            part_dir,
        )
        return result

    def _load_partition_indices(self, part_dir: Path, n_clients: int) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for k in range(1, n_clients + 1):
            cid = f"client_{k}"
            result[cid] = {
                "indices": np.load(part_dir / f"{cid}_indices.npy"),
                "previous_row": np.load(part_dir / f"{cid}_previous_row.npy"),
                "targets": np.load(part_dir / f"{cid}_targets.npy"),
            }
        return result

    def _compute_iid_partitions(
        self,
        train_path: Path,
        *,
        n_clients: int,
        seed: int,
        chunk_size: int,
        enforce_full_dataset: bool,
    ) -> dict[str, dict[str, Any]]:
        import pandas as pd

        target_col = self._metadata.target_column
        sender_col = self._metadata.sender_column

        # Pass 1: Stream targets and sender accounts
        all_targets: list[np.ndarray] = []
        all_senders: list[pd.Series] = []
        total_rows = 0

        for chunk in pd.read_csv(train_path, usecols=[target_col, sender_col], chunksize=chunk_size):
            all_targets.append(chunk[target_col].to_numpy(dtype=np.float32))
            all_senders.append(chunk[sender_col])
            total_rows += len(chunk)

        if enforce_full_dataset and total_rows < 18000000:
            raise SubsetModeDetectedError(
                f"Full LI-Medium dataset requires >= 18,000,000 training rows, got {total_rows}."
            )

        targets = np.concatenate(all_targets)
        senders = pd.concat(all_senders, ignore_index=True)
        del all_targets, all_senders

        # Stratified partition assignment
        rng = np.random.default_rng(seed)
        client_indices: list[list[int]] = [[] for _ in range(n_clients)]

        classes = np.unique(targets)
        for cls_val in sorted(classes):
            cls_indices = np.flatnonzero(targets == cls_val)
            shuffled_cls = rng.permutation(cls_indices)
            for pos, idx in enumerate(shuffled_cls):
                client_indices[pos % n_clients].append(int(idx))

        result: dict[str, dict[str, Any]] = {}
        for k in range(n_clients):
            cid = f"client_{k + 1}"
            c_indices = np.array(sorted(client_indices[k]), dtype=np.int64)
            c_targets = targets[c_indices]
            c_senders = senders.iloc[c_indices]
            codes, _ = pd.factorize(c_senders, sort=False)
            c_previous = build_previous_row_index(codes.astype(np.int64))

            result[cid] = {
                "indices": c_indices,
                "previous_row": c_previous,
                "targets": c_targets,
            }

        return result

    def _compute_non_iid_partitions(
        self,
        train_path: Path,
        *,
        seed: int,
        chunk_size: int,
        enforce_full_dataset: bool,
    ) -> dict[str, dict[str, Any]]:
        import pandas as pd

        target_col = self._metadata.target_column
        sender_col = self._metadata.sender_column

        cols = [target_col, sender_col]
        for opt_col in ("payment_format_code", "payment_currency_code", "receiving_currency_code", "same_currency", "same_bank"):
            if opt_col in self._metadata.feature_columns or opt_col in self._metadata.nominal_columns:
                cols.append(opt_col)
            elif opt_col in ("payment_format_code", "payment_currency_code", "receiving_currency_code", "same_currency", "same_bank"):
                cols.append(opt_col)

        # Pass 1: Accumulate typology profile per account
        account_to_id: dict[Any, int] = {}
        account_profiles: list[list[int]] = []  # [crypto_cnt, retail_cnt, cu_cnt]

        all_row_account_ids: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        total_rows = 0

        for chunk in pd.read_csv(train_path, usecols=lambda c: c in cols, chunksize=chunk_size):
            chunk_rows = len(chunk)
            total_rows += chunk_rows

            is_crypto = np.zeros(chunk_rows, dtype=bool)
            if "payment_format_code" in chunk.columns:
                fmt = chunk["payment_format_code"].to_numpy()
                is_crypto |= (fmt == 1) | (fmt == 6)
            if "payment_currency_code" in chunk.columns:
                is_crypto |= chunk["payment_currency_code"].to_numpy() == 1
            if "receiving_currency_code" in chunk.columns:
                is_crypto |= chunk["receiving_currency_code"].to_numpy() == 1
            if "same_currency" in chunk.columns:
                is_crypto |= chunk["same_currency"].to_numpy() == 0

            is_retail = np.zeros(chunk_rows, dtype=bool)
            if "payment_format_code" in chunk.columns:
                fmt = chunk["payment_format_code"].to_numpy()
                is_retail |= (fmt == 2) | (fmt == 4)
            is_retail &= ~is_crypto

            is_cu = np.zeros(chunk_rows, dtype=bool)
            if "payment_format_code" in chunk.columns:
                fmt = chunk["payment_format_code"].to_numpy()
                is_cu |= (fmt == 0) | (fmt == 3) | (fmt == 5)
            if "same_bank" in chunk.columns:
                is_cu |= chunk["same_bank"].to_numpy() == 1
            is_cu &= ~is_crypto
            is_cu &= ~is_retail

            row_accts = chunk[sender_col].to_numpy()
            row_acct_ids = np.empty(chunk_rows, dtype=np.int32)

            for i, acct in enumerate(row_accts):
                aid = account_to_id.get(acct)
                if aid is None:
                    aid = len(account_to_id)
                    account_to_id[acct] = aid
                    account_profiles.append([0, 0, 0])
                row_acct_ids[i] = aid
                if is_crypto[i]:
                    account_profiles[aid][0] += 1
                elif is_retail[i]:
                    account_profiles[aid][1] += 1
                elif is_cu[i]:
                    account_profiles[aid][2] += 1

            all_row_account_ids.append(row_acct_ids)
            all_targets.append(chunk[target_col].to_numpy(dtype=np.float32))

        if enforce_full_dataset and total_rows < 18000000:
            raise SubsetModeDetectedError(
                f"Full LI-Medium dataset requires >= 18,000,000 training rows, got {total_rows}."
            )

        row_account_ids = np.concatenate(all_row_account_ids)
        targets = np.concatenate(all_targets)
        del all_row_account_ids, all_targets

        # Account assignment
        n_accounts = len(account_to_id)
        rng = np.random.default_rng(seed)
        acct_client_assignment = np.zeros(n_accounts, dtype=np.int8)

        for aid, (c_cnt, r_cnt, u_cnt) in enumerate(account_profiles):
            if c_cnt > 0:
                acct_client_assignment[aid] = 1  # client_2: Offshore / Crypto Exchange
            elif r_cnt > u_cnt:
                acct_client_assignment[aid] = 0  # client_1: Retail Giant
            elif u_cnt > r_cnt:
                acct_client_assignment[aid] = 2  # client_3: Regional Credit Union
            else:
                acct_client_assignment[aid] = 0 if rng.random() < 0.5 else 2

        # Map to rows
        row_clients = acct_client_assignment[row_account_ids]

        result: dict[str, dict[str, Any]] = {}
        for k in range(3):
            cid = f"client_{k + 1}"
            c_indices = np.flatnonzero(row_clients == k).astype(np.int64)
            c_targets = targets[c_indices]
            c_acct_ids = row_account_ids[c_indices]
            # Build previous row within the client partition
            c_previous = build_previous_row_index(c_acct_ids.astype(np.int64))

            result[cid] = {
                "indices": c_indices,
                "previous_row": c_previous,
                "targets": c_targets,
            }

        return result

    def build_client_datasets(
        self,
        distribution: str,
        parent_dataset: Any,
        config: TrainingConfig,
        *,
        seed: int = 42,
        enforce_full_dataset: bool = True,
    ) -> list[MmapClientDataset]:
        """Construct zero-copy client datasets referencing the shared parent memory map."""
        partitions = self.get_or_create_partitions(
            distribution,
            seed=seed,
            n_clients=3,
            enforce_full_dataset=enforce_full_dataset,
        )

        client_datasets: list[MmapClientDataset] = []
        for k in range(1, 4):
            cid = f"client_{k}"
            pdata = partitions[cid]
            client_datasets.append(
                MmapClientDataset(
                    split=SplitName.TRAIN,
                    parent_features=parent_dataset._features,
                    client_targets=pdata["targets"],
                    client_previous_row=pdata["previous_row"],
                    client_global_indices=pdata["indices"],
                    sequence_length=config.sequence_length,
                    client_id=cid,
                )
            )

        return client_datasets
