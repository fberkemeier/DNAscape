"""Explicit session-state helpers for the DNAscape app.

Only keys registered through :func:`persistent_widget_key` are carried across
router page changes. Uploaded bytes and processed datasets live in canonical
registries, so pages do not depend on Streamlit retaining an inactive
``file_uploader`` widget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping

import streamlit as st

from app_streamlit.workflow_models import scientific_signature

PERSISTENT_WIDGET_KEYS = "dnascape_persistent_widget_keys_v2"
UPLOADS_KEY = "dnascape_upload_registry"
DATASETS_KEY = "dnascape_dataset_registry"
WORKFLOWS_KEY = "dnascape_workflow_state"
UPLOAD_WIDGET_TOKENS_KEY = "dnascape_upload_widget_tokens"
INSPECTIONS_KEY = "dnascape_source_inspections"
DATASET_SUMMARIES_KEY = "dnascape_dataset_summaries"
MAX_RETAINED_UPLOADS = 32
MAX_RETAINED_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_SINGLE_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_RETAINED_DATASETS = 32
MAX_RETAINED_DATASET_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class UploadRecord:
    """One content-addressed upload retained once for the current session."""

    digest: str
    name: str
    payload: bytes
    mime: str = "application/octet-stream"
    created_at: float = field(default_factory=time.time)

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass
class StoredWorkflowResult:
    """A scientific result together with the inputs that make it valid."""

    payload: Any
    input_signature: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


def _mapping(key: str) -> dict[str, Any]:
    value = st.session_state.get(key)
    if not isinstance(value, dict):
        value = {}
        st.session_state[key] = value
    return value


def persistent_widget_key(namespace: str, name: str) -> str:
    """Return and register a stable, page-prefixed Streamlit widget key."""

    namespace = str(namespace).strip().replace(" ", "_")
    name = str(name).strip().replace(" ", "_")
    key = f"{namespace}_{name}"
    registered = st.session_state.get(PERSISTENT_WIDGET_KEYS)
    if not isinstance(registered, list):
        registered = []
    if key not in registered:
        registered.append(key)
        st.session_state[PERSISTENT_WIDGET_KEYS] = registered
    return key


def initialized_widget_key(namespace: str, name: str, default: Any) -> str:
    """Register a persistent key and initialize it before widget creation.

    Streamlit warns when a widget receives both ``value=`` and an explicitly
    preserved session-state value.  Initializing the key once and omitting the
    widget default avoids that warning on every router rerun.
    """

    key = persistent_widget_key(namespace, name)
    if key not in st.session_state:
        st.session_state[key] = default
    return key


def preserve_persistent_widget_state() -> None:
    """Interrupt Streamlit cleanup only for explicitly registered controls."""

    registered = st.session_state.get(PERSISTENT_WIDGET_KEYS, [])
    if not isinstance(registered, (list, tuple, set)):
        return
    for key in registered:
        key = str(key)
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def store_upload(
    name: str,
    payload: bytes | bytearray | memoryview,
    mime: str = "application/octet-stream",
) -> UploadRecord:
    """Store uploaded content once and return its content-addressed record."""

    payload_size = memoryview(payload).nbytes
    if payload_size > MAX_SINGLE_UPLOAD_BYTES:
        raise ValueError(
            "The upload exceeds the app's 256 MiB per-file safety limit."
        )
    payload_bytes = bytes(payload)
    digest = sha256(payload_bytes).hexdigest()
    registry = _mapping(UPLOADS_KEY)
    existing = registry.get(digest)
    if isinstance(existing, UploadRecord):
        return existing
    record = UploadRecord(
        digest=digest,
        name=str(name) or "uploaded-data",
        payload=payload_bytes,
        mime=str(mime or "application/octet-stream"),
    )
    registry[digest] = record
    while (
        len(registry) > MAX_RETAINED_UPLOADS
        or sum(
            item.size_bytes
            for item in registry.values()
            if isinstance(item, UploadRecord)
        )
        > MAX_RETAINED_UPLOAD_BYTES
    ):
        candidates = [
            item
            for item in registry.values()
            if isinstance(item, UploadRecord) and item.digest != digest
        ]
        if not candidates:
            registry.pop(digest, None)
            raise ValueError(
                "Retaining this upload would exceed the app's 512 MiB session limit."
            )
        oldest = min(candidates, key=lambda item: item.created_at)
        remove_upload(oldest.digest)
    return record


def resolve_uploaded_file(uploaded_file: Any, widget_key: str) -> UploadRecord | None:
    """Resolve an UploadedFile without re-reading or hashing unchanged content."""

    token_registry = _mapping(UPLOAD_WIDGET_TOKENS_KEY)
    prior = token_registry.get(str(widget_key))
    if uploaded_file is None:
        if isinstance(prior, dict):
            return get_upload(str(prior.get("digest") or ""))
        return None
    name = str(getattr(uploaded_file, "name", "uploaded-data"))
    size = int(getattr(uploaded_file, "size", 0) or 0)
    mime = str(getattr(uploaded_file, "type", "application/octet-stream") or "")
    file_id = getattr(uploaded_file, "file_id", None)
    token = f"{file_id or id(uploaded_file)}:{name}:{size}:{mime}"
    if isinstance(prior, dict) and prior.get("token") == token:
        retained = get_upload(str(prior.get("digest") or ""))
        if retained is not None:
            return retained
    retained = store_upload(name, uploaded_file.getvalue(), mime)
    token_registry[str(widget_key)] = {"token": token, "digest": retained.digest}
    return retained


def forget_uploaded_file(widget_key: str) -> None:
    """Forget one uploader's retained selection without deleting shared bytes."""

    _mapping(UPLOAD_WIDGET_TOKENS_KEY).pop(str(widget_key), None)


def uploads() -> tuple[UploadRecord, ...]:
    """Return uploads without copying their byte payloads."""

    return tuple(
        item for item in _mapping(UPLOADS_KEY).values() if isinstance(item, UploadRecord)
    )


def get_upload(digest: str | None) -> UploadRecord | None:
    """Return one retained upload by digest."""

    if not digest:
        return None
    value = _mapping(UPLOADS_KEY).get(str(digest))
    return value if isinstance(value, UploadRecord) else None


def remove_upload(digest: str) -> None:
    """Remove one upload; processed datasets remain independently usable."""

    digest = str(digest)
    _mapping(UPLOADS_KEY).pop(digest, None)
    _mapping(INSPECTIONS_KEY).pop(digest, None)
    token_registry = _mapping(UPLOAD_WIDGET_TOKENS_KEY)
    for key, value in tuple(token_registry.items()):
        if isinstance(value, Mapping) and str(value.get("digest")) == digest:
            token_registry.pop(key, None)


def get_source_inspection(digest: str | None) -> Any | None:
    """Return cached source metadata by content digest."""

    if not digest:
        return None
    return _mapping(INSPECTIONS_KEY).get(str(digest))


def set_source_inspection(digest: str, inspection: Any) -> Any:
    """Retain lightweight inspection metadata, never a parsed data copy."""

    _mapping(INSPECTIONS_KEY)[str(digest)] = inspection
    return inspection


def register_dataset(record: Any) -> Any:
    """Register one canonical dataset object by its ``dataset_id`` attribute."""

    dataset_id = getattr(record, "dataset_id", None)
    if not dataset_id and isinstance(record, Mapping):
        dataset_id = record.get("dataset_id")
    if not dataset_id:
        raise ValueError("A registered dataset must have a non-empty dataset_id.")
    registry = _mapping(DATASETS_KEY)
    dataset_id = str(dataset_id)
    if dataset_id in registry:
        registry[dataset_id] = record
        return record
    try:
        record_bytes = max(0, int(getattr(record, "nbytes", 0)))
    except (TypeError, ValueError, OverflowError):
        record_bytes = 0
    retained_bytes = 0
    for item in registry.values():
        try:
            retained_bytes += max(0, int(getattr(item, "nbytes", 0)))
        except (TypeError, ValueError, OverflowError):
            continue
    if len(registry) >= MAX_RETAINED_DATASETS:
        raise MemoryError(
            f"The session already contains {MAX_RETAINED_DATASETS} prepared datasets. "
            "Remove one on Cached Files before preparing another."
        )
    if retained_bytes + record_bytes > MAX_RETAINED_DATASET_BYTES:
        raise MemoryError(
            "Retaining this dataset would exceed the app's 1 GiB prepared-data limit. "
            "Remove unused datasets or prepare a smaller interval."
        )
    registry[dataset_id] = record
    return record


def datasets() -> tuple[Any, ...]:
    """Return canonical datasets without copying contained arrays."""

    return tuple(_mapping(DATASETS_KEY).values())


def get_dataset(dataset_id: str | None) -> Any | None:
    """Return one canonical dataset by id."""

    if not dataset_id:
        return None
    return _mapping(DATASETS_KEY).get(str(dataset_id))


def remove_dataset(dataset_id: str) -> None:
    """Remove one canonical dataset from this session."""

    dataset_id = str(dataset_id)
    _mapping(DATASETS_KEY).pop(dataset_id, None)
    _mapping(DATASET_SUMMARIES_KEY).pop(dataset_id, None)


def get_dataset_summary(dataset_id: str | None) -> Any | None:
    """Return a cached compact summary for a canonical dataset."""

    if not dataset_id:
        return None
    return _mapping(DATASET_SUMMARIES_KEY).get(str(dataset_id))


def set_dataset_summary(dataset_id: str, summary: Any) -> Any:
    """Retain one small numerical summary by canonical dataset id."""

    _mapping(DATASET_SUMMARIES_KEY)[str(dataset_id)] = summary
    return summary


def workflow_input_signature(*parts: Any) -> str:
    """Return the shared deterministic scientific signature."""

    return scientific_signature(*parts)


def _workflow_bucket(workflow: str) -> dict[str, Any]:
    workflows = _mapping(WORKFLOWS_KEY)
    workflow = str(workflow)
    bucket = workflows.get(workflow)
    if not isinstance(bucket, dict):
        bucket = {}
        workflows[workflow] = bucket
    return bucket


def set_workflow_draft(workflow: str, draft: Mapping[str, Any]) -> None:
    """Persist lightweight draft controls separately from scientific results."""

    _workflow_bucket(workflow)["draft"] = dict(draft)


def get_workflow_draft(workflow: str) -> dict[str, Any]:
    """Return a copy of a workflow's lightweight draft settings."""

    draft = _workflow_bucket(workflow).get("draft", {})
    return dict(draft) if isinstance(draft, Mapping) else {}


def set_workflow_result(
    workflow: str,
    result: Any,
    input_signature: str,
    metadata: Mapping[str, Any] | None = None,
) -> StoredWorkflowResult:
    """Store a result and the exact scientific signature that produced it."""

    _clear_workflow_summaries(workflow)
    stored = StoredWorkflowResult(
        payload=result,
        input_signature=str(input_signature),
        metadata=dict(metadata or {}),
    )
    _workflow_bucket(workflow)["result"] = stored
    return stored


def get_workflow_result(workflow: str) -> StoredWorkflowResult | None:
    """Return the last stored result for a workflow."""

    value = _workflow_bucket(workflow).get("result")
    return value if isinstance(value, StoredWorkflowResult) else None


def workflow_result_is_stale(workflow: str, current_signature: str) -> bool:
    """Return whether a stored result was produced from different inputs."""

    result = get_workflow_result(workflow)
    return result is not None and result.input_signature != str(current_signature)


def clear_workflow_result(workflow: str) -> None:
    """Discard a workflow result without resetting its draft controls."""

    _workflow_bucket(workflow).pop("result", None)
    _clear_workflow_summaries(workflow)


def _clear_workflow_summaries(workflow: str) -> None:
    """Discard small derived summaries when their workflow payload changes."""

    prefix = f"workflow:{str(workflow)}:"
    summaries = _mapping(DATASET_SUMMARIES_KEY)
    for key in tuple(summaries):
        if str(key).startswith(prefix):
            summaries.pop(key, None)


def workflow_results() -> dict[str, StoredWorkflowResult]:
    """Return stored workflow results without copying their payloads."""

    results: dict[str, StoredWorkflowResult] = {}
    for name, bucket in _mapping(WORKFLOWS_KEY).items():
        if isinstance(bucket, Mapping):
            result = bucket.get("result")
            if isinstance(result, StoredWorkflowResult):
                results[str(name)] = result
    return results


def harvest_completed_workflow_jobs() -> tuple[str, ...]:
    """Publish finished background results from any active page rerun.

    A user may start a long calculation and navigate elsewhere.  Polling the
    two known job slots in the shared shell makes the completed result
    immediately available to Plotting and Cached Files without requiring a
    return visit to the originating page.
    """

    from app_streamlit.jobs import BackgroundJob, poll_background_job

    completed: list[str] = []
    job_specs = {
        "mapping": (
            "mapping_background_job_v2",
            ("mapping_plot_preview_v2", "mapping_raw_export_v2", "mapping_figure_export_v2"),
        ),
        "simulation": (
            "simulation_background_job_v2",
            ("simulation_plot_preview_v2", "simulation_raw_export_v2", "simulation_figure_export_v2"),
        ),
    }
    for workflow, (job_key, derived_keys) in job_specs.items():
        job = st.session_state.get(job_key)
        if not isinstance(job, BackgroundJob):
            continue
        poll_background_job(job)
        if job.status != "complete" or job.consumed:
            continue
        payload = job.result
        if not isinstance(payload, Mapping):
            job.status = "error"
            job.error_type = "InvalidJobResult"
            job.error_message = (
                f"{workflow.title()} returned an invalid result instead of a mapping."
            )
            job.result = None
            continue
        metadata: dict[str, Any] = {
            "elapsed_seconds": payload.get("elapsed_seconds")
        }
        if workflow == "mapping":
            metadata.update(
                route=payload.get("route_label", job.label),
                method=payload.get("method", "rmap"),
            )
        else:
            metadata.update(
                realizations=payload.get("realizations"),
                seed=payload.get("seed"),
            )
        set_workflow_result(workflow, payload, job.input_signature, metadata)
        job.consumed = True
        # The registry now owns the payload.  Do not leave a second hidden
        # reference on the completed job after a user clears the result.
        job.result = None
        for key in derived_keys:
            st.session_state.pop(key, None)
        completed.append(workflow)
    return tuple(completed)
