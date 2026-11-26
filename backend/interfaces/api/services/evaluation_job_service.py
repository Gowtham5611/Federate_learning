"""Background job execution and result persistence service for CSV evaluations.

Eliminates cloud reverse-proxy timeouts (HTTP 502/504) by decoupling upload
acceptance from heavy sequence preprocessing and model inference.
Provides:
- Immediate 202 Accepted response with job ID.
- Idempotent deduplication based on content hash.
- Thread-safe background execution with granular progress tracking.
- Disk-backed result persistence for state recovery on page reload.
- Non-sensitive diagnostic logging.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[3]
_ROOT_DIR = _BACKEND_DIR.parent
BASE_DIR = _BACKEND_DIR if (_BACKEND_DIR / "data").exists() else _ROOT_DIR


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    run_id: str
    status: JobStatus
    progress: float
    phase: str
    created_at: float
    completed_at: Optional[float] = None
    filename: str = "upload.csv"
    dataset: Optional[str] = None
    model_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_phase: Optional[str] = None
    content_hash: Optional[str] = None


class EvaluationJobService:
    """Singleton service managing asynchronous CSV evaluation jobs."""

    _instance: Optional["EvaluationJobService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "EvaluationJobService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_service()
            return cls._instance

    def _init_service(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._hash_to_job: Dict[str, str] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="aml-eval-worker",
        )
        self._state_lock = threading.Lock()

        # Resolve persistent results directory
        candidate_dir = BASE_DIR / "data/results_store"
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            self._storage_dir = candidate_dir
        except Exception:
            fallback = Path(tempfile.gettempdir()) / "fedsecureai_results_store"
            fallback.mkdir(parents=True, exist_ok=True)
            self._storage_dir = fallback

        logger.info("EvaluationJobService initialized with storage at %s", self._storage_dir)

    def _persist_job(self, job: JobRecord) -> None:
        """Persist completed or failed job record to disk for reload resilience."""
        try:
            file_path = self._storage_dir / f"{job.job_id}.json"
            data = {
                "job_id": job.job_id,
                "run_id": job.run_id,
                "status": job.status.value,
                "progress": job.progress,
                "phase": job.phase,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "filename": job.filename,
                "dataset": job.dataset,
                "model_id": job.model_id,
                "error": job.error,
                "error_phase": job.error_phase,
                "result": job.result,
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Failed to persist job %s to disk: %s", job.job_id, e)

    def _load_persisted_job(self, job_id: str) -> Optional[JobRecord]:
        """Attempt to restore job record from disk."""
        file_path = self._storage_dir / f"{job_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            job = JobRecord(
                job_id=data["job_id"],
                run_id=data.get("run_id", data["job_id"]),
                status=JobStatus(data["status"]),
                progress=data.get("progress", 1.0),
                phase=data.get("phase", "done"),
                created_at=data.get("created_at", time.time()),
                completed_at=data.get("completed_at"),
                filename=data.get("filename", "upload.csv"),
                dataset=data.get("dataset"),
                model_id=data.get("model_id"),
                error=data.get("error"),
                error_phase=data.get("error_phase"),
                result=data.get("result"),
            )
            with self._state_lock:
                self._jobs[job_id] = job
            return job
        except Exception as e:
            logger.warning("Failed to load persisted job %s: %s", job_id, e)
            return None

    def submit_job(
        self,
        csv_bytes: bytes,
        filename: str = "upload.csv",
        dataset: Optional[str] = None,
        model_id: Optional[str] = None,
        row_limit: Optional[int] = None,
        threshold: Optional[float] = None,
        user_mappings: Optional[Dict[str, str]] = None,
        tenant_id: str = "default_tenant",
        enable_persistent_history: bool = False,
        fallback_model: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Submit a CSV evaluation job.

        Returns:
            Tuple of (job_id, is_new_job). If an identical evaluation is already
            in progress or recently completed, returns the existing job ID.
        """
        # Deduplication hash: sha256(content + model + threshold)
        hash_seed = csv_bytes + (model_id or "").encode("utf-8") + str(threshold or "").encode("utf-8")
        content_hash = hashlib.sha256(hash_seed).hexdigest()

        with self._state_lock:
            if content_hash in self._hash_to_job:
                existing_id = self._hash_to_job[content_hash]
                if existing_id in self._jobs:
                    existing = self._jobs[existing_id]
                    if existing.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                        logger.info("Deduplicating submission: returning active job %s", existing_id)
                        return existing_id, False
                    if existing.status == JobStatus.COMPLETE:
                        # Return cached result if finished within last 15 minutes
                        if existing.completed_at and (time.time() - existing.completed_at < 900):
                            logger.info("Deduplicating submission: returning cached result %s", existing_id)
                            return existing_id, False

            job_id = f"job_{uuid.uuid4().hex[:12]}"
            run_id = job_id

            job = JobRecord(
                job_id=job_id,
                run_id=run_id,
                status=JobStatus.QUEUED,
                progress=0.05,
                phase="queued",
                created_at=time.time(),
                filename=filename,
                dataset=dataset,
                model_id=model_id,
                content_hash=content_hash,
            )

            self._jobs[job_id] = job
            self._hash_to_job[content_hash] = job_id

        # Schedule execution in background thread
        self._executor.submit(
            self._execute_job,
            job_id=job_id,
            csv_bytes=csv_bytes,
            filename=filename,
            dataset=dataset,
            model_id=model_id,
            row_limit=row_limit,
            threshold=threshold,
            user_mappings=user_mappings,
            tenant_id=tenant_id,
            enable_persistent_history=enable_persistent_history,
            fallback_model=fallback_model,
        )

        return job_id, True

    def _execute_job(
        self,
        job_id: str,
        csv_bytes: bytes,
        filename: str,
        dataset: Optional[str],
        model_id: Optional[str],
        row_limit: Optional[int],
        threshold: Optional[float],
        user_mappings: Optional[Dict[str, str]],
        tenant_id: str,
        enable_persistent_history: bool,
        fallback_model: Optional[str],
    ) -> None:
        """Worker function executing the end-to-end evaluation with phase updates."""
        try:
            from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
        except ModuleNotFoundError:
            from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService

        logger.info("Beginning execution of evaluation job %s for '%s'", job_id, filename)

        with self._state_lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].status = JobStatus.RUNNING
            self._jobs[job_id].progress = 0.20
            self._jobs[job_id].phase = "preprocessing_and_schema_adaptation"

        try:
            service = CsvEvaluationService()

            with self._state_lock:
                self._jobs[job_id].progress = 0.40
                self._jobs[job_id].phase = "sequence_construction"

            def _update_progress(prog: float, phase_name: str) -> None:
                with self._state_lock:
                    if job_id in self._jobs:
                        self._jobs[job_id].progress = min(0.89, max(0.05, float(prog)))
                        self._jobs[job_id].phase = phase_name

            # Execute evaluation
            result = service.evaluate_csv_bytes(
                csv_bytes=csv_bytes,
                dataset=dataset,
                filename=filename,
                row_limit=row_limit,
                model_id=model_id or "auto",
                user_mappings=user_mappings,
                tenant_id=tenant_id,
                enable_persistent_history=enable_persistent_history,
                threshold=threshold,
                fallback_model=fallback_model,
                progress_callback=_update_progress,
            )

            with self._state_lock:
                self._jobs[job_id].progress = 0.90
                self._jobs[job_id].phase = "finalizing_metrics"

                # Link run ID and download ID to job
                result["job_id"] = job_id
                result["run_id"] = job_id
                if not result.get("prediction_id"):
                    result["prediction_id"] = job_id

                self._jobs[job_id].status = JobStatus.COMPLETE
                self._jobs[job_id].progress = 1.0
                self._jobs[job_id].phase = "complete"
                self._jobs[job_id].completed_at = time.time()
                self._jobs[job_id].result = result

            self._persist_job(self._jobs[job_id])
            logger.info("Evaluation job %s completed successfully in %.2fs", job_id, time.time() - self._jobs[job_id].created_at)

        except ValueError as ve:
            logger.warning("Validation failure in evaluation job %s: %s", job_id, ve)
            with self._state_lock:
                self._jobs[job_id].status = JobStatus.FAILED
                self._jobs[job_id].progress = 1.0
                self._jobs[job_id].phase = "validation_error"
                self._jobs[job_id].error_phase = "validation"
                self._jobs[job_id].error = str(ve)
                self._jobs[job_id].completed_at = time.time()
            self._persist_job(self._jobs[job_id])

        except Exception as e:
            logger.error("Internal failure in evaluation job %s: %s", job_id, e, exc_info=True)
            with self._state_lock:
                self._jobs[job_id].status = JobStatus.FAILED
                self._jobs[job_id].progress = 1.0
                self._jobs[job_id].phase = "execution_error"
                self._jobs[job_id].error_phase = "inference"
                self._jobs[job_id].error = f"Model evaluation encountered an unexpected error: {str(e)}"
                self._jobs[job_id].completed_at = time.time()
            self._persist_job(self._jobs[job_id])

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job status, progress, and result."""
        with self._state_lock:
            job = self._jobs.get(job_id)

        if not job:
            job = self._load_persisted_job(job_id)

        if not job:
            return None

        return {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "status": job.status.value,
            "progress": job.progress,
            "phase": job.phase,
            "filename": job.filename,
            "dataset": job.dataset,
            "model_id": job.model_id,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "result": job.result if job.status == JobStatus.COMPLETE else None,
            "error": job.error if job.status == JobStatus.FAILED else None,
            "error_phase": job.error_phase if job.status == JobStatus.FAILED else None,
        }

    def list_recent_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recently executed evaluation jobs."""
        with self._state_lock:
            records = list(self._jobs.values())

        records.sort(key=lambda r: r.created_at, reverse=True)
        return [
            {
                "job_id": r.job_id,
                "run_id": r.run_id,
                "status": r.status.value,
                "progress": r.progress,
                "phase": r.phase,
                "filename": r.filename,
                "dataset": r.dataset,
                "created_at": r.created_at,
                "completed_at": r.completed_at,
                "has_result": r.result is not None,
                "error": r.error,
            }
            for r in records[:limit]
        ]
