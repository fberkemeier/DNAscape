"""Session storage inventory for uploads, datasets, results, and exports."""

from __future__ import annotations

import gzip
import html
import math
from datetime import datetime
from functools import partial
from typing import Any, Mapping, Sequence

import numpy as np
import streamlit as st

from app_streamlit.app_shell import (
    GENERATED_ARTIFACTS_KEY,
    MAX_GENERATED_CACHE_BYTES_LIMIT,
    MAX_GENERATED_CACHE_LIMIT,
    artifact_size_label,
    clear_generated_artifacts,
    generated_artifact_size,
    generated_artifacts,
    generated_cache_bytes_limit,
    generated_cache_keep_plots,
    generated_cache_keep_raw,
    generated_cache_limit,
    remove_generated_artifact,
    render_page_intro,
    render_standard_content_marker,
    set_generated_cache_bytes_limit,
    set_generated_cache_keep_plots,
    set_generated_cache_keep_raw,
    set_generated_cache_limit,
)
from app_streamlit.data_service import DatasetRecord, dataset_display_name
from app_streamlit.preview import memory_label
from app_streamlit.state import (
    clear_workflow_result,
    datasets,
    persistent_widget_key,
    remove_dataset,
    remove_upload,
    uploads,
    workflow_results,
)

PAGE_NAMESPACE = "cached_files"
GZIP_PREVIEW_KEY = "cached_files_prepared_gzip"


def _cached_bytes(value: object) -> bytes | None:
    """Return a safe immutable cache payload without accepting arbitrary objects."""

    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return None


def _sanitize_generated_artifact_state() -> int:
    """Repair legacy or damaged cache entries before shared pruning traverses them.

    Generated artifacts normally enter through ``register_generated_artifact``.
    Session state can nevertheless outlive code revisions, and browser clients can
    replay stale widget/session data.  Keep every recoverable byte payload, assign
    unique stable IDs, and discard entries that cannot be downloaded safely.
    """

    raw_artifacts = st.session_state.get(GENERATED_ARTIFACTS_KEY)
    if not isinstance(raw_artifacts, list):
        return 0

    sanitized: list[dict[str, object]] = []
    used_ids: set[str] = set()
    discarded = 0
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, dict):
            discarded += 1
            continue

        fallback_name = str(raw_artifact.get("file_name") or "dnascape-output.bin")
        fallback_mime = str(
            raw_artifact.get("mime") or "application/octet-stream"
        )
        normalized_variants: dict[str, dict[str, object]] = {}
        raw_variants = raw_artifact.get("variants")
        if isinstance(raw_variants, Mapping):
            for raw_label, raw_variant in raw_variants.items():
                if not isinstance(raw_variant, Mapping):
                    continue
                payload = _cached_bytes(raw_variant.get("payload"))
                if payload is None:
                    continue
                label = str(raw_variant.get("label") or raw_label).strip() or "File"
                normalized_variants[label] = {
                    "label": label,
                    "file_name": str(raw_variant.get("file_name") or fallback_name),
                    "mime": str(raw_variant.get("mime") or fallback_mime),
                    "payload": payload,
                    "size": len(payload),
                }

        fallback_payload = _cached_bytes(raw_artifact.get("payload"))
        if not normalized_variants and fallback_payload is not None:
            normalized_variants["File"] = {
                "label": "File",
                "file_name": fallback_name,
                "mime": fallback_mime,
                "payload": fallback_payload,
                "size": len(fallback_payload),
            }
        if not normalized_variants:
            discarded += 1
            continue

        base_id = str(raw_artifact.get("id") or f"legacy-{index}").strip()
        base_id = base_id or f"legacy-{index}"
        artifact_id = base_id
        suffix = 2
        while artifact_id in used_ids:
            artifact_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(artifact_id)

        first_variant = next(iter(normalized_variants.values()))
        normalized = dict(raw_artifact)
        normalized.update(
            id=artifact_id,
            file_name=fallback_name,
            mime=fallback_mime,
            payload=first_variant["payload"],
            size=sum(
                len(_cached_bytes(variant.get("payload")) or b"")
                for variant in normalized_variants.values()
            ),
            variants=normalized_variants,
        )
        sanitized.append(normalized)

    st.session_state[GENERATED_ARTIFACTS_KEY] = sanitized
    return discarded


def artifact_download_variants(
    artifact: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Return download variants without copying their immutable payload bytes."""

    variants = artifact.get("variants")
    if isinstance(variants, dict) and variants:
        return {
            str(label): variant
            for label, variant in variants.items()
            if isinstance(variant, dict) and variant.get("payload") is not None
        }
    if artifact.get("payload") is None:
        return {}
    return {
        "File": {
            "file_name": artifact.get("file_name", "dnascape-output.bin"),
            "mime": artifact.get("mime", "application/octet-stream"),
            "payload": artifact["payload"],
        }
    }


def selected_download_payload(
    variant: dict[str, object],
    *,
    compress: bool,
) -> tuple[bytes, str, str]:
    """Return direct bytes, compressing only when explicitly requested."""

    raw_payload = variant.get("payload", b"")
    if isinstance(raw_payload, bytes):
        payload = raw_payload
    elif isinstance(raw_payload, (bytearray, memoryview)):
        payload = bytes(raw_payload)
    else:
        raise ValueError("The cached download payload is not bytes-like.")
    file_name = str(variant.get("file_name") or "dnascape-output.bin")
    mime = str(variant.get("mime") or "application/octet-stream")
    if not compress:
        return payload, file_name, mime
    return gzip.compress(payload), f"{file_name}.gz", "application/gzip"


def _compressed_download_bytes(variant: dict[str, object]) -> bytes:
    """Compress a selected variant only when its download is requested."""

    return selected_download_payload(variant, compress=True)[0]


def _timestamp(value: object) -> float:
    """Return a safe artifact timestamp for display and sorting."""

    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _display_datetime(value: object) -> datetime:
    """Convert a retained timestamp without exposing platform range errors."""

    try:
        return datetime.fromtimestamp(_timestamp(value))
    except (OverflowError, OSError, ValueError):
        return datetime.fromtimestamp(0.0)


def sorted_generated_artifacts(
    artifacts: list[dict[str, object]],
    sort_key: str,
    descending: bool,
) -> list[dict[str, object]]:
    """Sort generated artifacts without touching their payloads."""

    if sort_key == "size":
        key = lambda artifact: (
            generated_artifact_size(artifact),
            _timestamp(artifact.get("created_at", 0.0)),
        )
    else:
        key = lambda artifact: (
            _timestamp(artifact.get("created_at", 0.0)),
            generated_artifact_size(artifact),
        )
    return sorted(artifacts, key=key, reverse=descending)


def _payload_nbytes(value: Any, seen: set[int] | None = None, depth: int = 0) -> int:
    """Estimate retained workflow bytes without converting or traversing deeply."""

    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen or depth > 4:
        return 0
    seen.add(object_id)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_payload_nbytes(item, seen, depth + 1) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_payload_nbytes(item, seen, depth + 1) for item in value)
    return 0


def _page_slice(items: Sequence[Any], namespace: str) -> tuple[Sequence[Any], int, int]:
    """Render compact pagination controls and return the current page slice."""

    if not items:
        return items, 1, 1
    size_key = persistent_widget_key(PAGE_NAMESPACE, f"{namespace}_page_size")
    retained_size = st.session_state.get(size_key)
    if not isinstance(retained_size, int) or retained_size not in {10, 25, 50}:
        st.session_state[size_key] = 10
    page_size = st.selectbox("Rows per page", (10, 25, 50), key=size_key)
    pages = max(1, math.ceil(len(items) / int(page_size)))
    page_key = persistent_widget_key(PAGE_NAMESPACE, f"{namespace}_page")
    retained_page = st.session_state.get(page_key)
    if not isinstance(retained_page, int):
        retained_page = 1
    st.session_state[page_key] = min(max(1, retained_page), pages)
    page = int(
        st.number_input(
            "Page",
            min_value=1,
            max_value=pages,
            step=1,
            key=page_key,
        )
    )
    start = (page - 1) * int(page_size)
    return items[start : start + int(page_size)], page, pages


def _rerun() -> None:
    st.rerun()


def _render_settings() -> None:
    with st.container(border=True):
        st.markdown("#### Cache settings")
        columns = st.columns((0.17, 0.17, 0.18, 0.18, 0.3), gap="small")
        with columns[0]:
            limit = st.number_input(
                "Max generated files",
                min_value=1,
                max_value=MAX_GENERATED_CACHE_LIMIT,
                value=generated_cache_limit(),
                step=1,
                key="cached_files_limit",
            )
            if int(limit) != generated_cache_limit():
                set_generated_cache_limit(limit)
        with columns[1]:
            current_mib = generated_cache_bytes_limit() // (1024 * 1024)
            budget_mib = st.number_input(
                "Max generated MiB",
                min_value=8,
                max_value=MAX_GENERATED_CACHE_BYTES_LIMIT // (1024 * 1024),
                value=current_mib,
                step=8,
                key="cached_files_byte_limit",
                help="Oldest generated artifacts are pruned when this byte budget is exceeded.",
            )
            if int(budget_mib) != current_mib:
                set_generated_cache_bytes_limit(int(budget_mib) * 1024 * 1024)
        with columns[2]:
            keep_raw = st.checkbox(
                "Retain new raw exports",
                value=generated_cache_keep_raw(),
                key="cached_files_keep_raw",
                help="Disabled by default to avoid retaining large redundant serializations.",
            )
            set_generated_cache_keep_raw(keep_raw)
        with columns[3]:
            keep_plots = st.checkbox(
                "Retain new plot exports",
                value=generated_cache_keep_plots(),
                key="cached_files_keep_plots",
            )
            set_generated_cache_keep_plots(keep_plots)
        with columns[4]:
            st.caption(
                "Uploads and canonical datasets are managed separately. Removing an upload does not "
                "delete an already prepared array."
            )


def _render_generated() -> None:
    artifacts = generated_artifacts()
    controls = st.columns((0.35, 0.35, 0.3), gap="small")
    sort_key = controls[0].selectbox(
        "Sort by",
        ("date", "size"),
        key=persistent_widget_key(PAGE_NAMESPACE, "generated_sort"),
    )
    descending = controls[1].checkbox(
        "Newest / largest first",
        value=True,
        key=persistent_widget_key(PAGE_NAMESPACE, "generated_descending"),
    )
    if controls[2].button(
        "Clear generated files",
        disabled=not artifacts,
        width="stretch",
        key="cached_files_clear_generated",
    ):
        clear_generated_artifacts()
        st.session_state.pop(GZIP_PREVIEW_KEY, None)
        _rerun()
    artifacts = sorted_generated_artifacts(artifacts, sort_key, descending)
    total = sum(generated_artifact_size(item) for item in artifacts)
    st.caption(f"{len(artifacts)} generated file(s) - {artifact_size_label(total)} retained")
    shown, page, pages = _page_slice(artifacts, "generated")
    if not shown:
        st.info("No generated exports are retained in this session.")
        return
    st.caption(f"Page {page} of {pages}")
    for _index, artifact in enumerate(shown):
        artifact_id = str(artifact.get("id", ""))
        variants = artifact_download_variants(artifact)
        if not variants:
            continue
        labels = list(variants)
        with st.container(border=True):
            heading, meta, actions = st.columns((0.47, 0.25, 0.28), gap="small")
            with heading:
                st.write(str(artifact.get("file_name") or "DNAscape output"))
                st.caption(
                    f"{artifact.get('source', 'DNAscape')} - {artifact.get('category', 'Output')} - "
                    f"{artifact_size_label(generated_artifact_size(artifact))}"
                )
            with meta:
                created = _display_datetime(artifact.get("created_at", 0.0))
                st.caption(created.strftime("%Y-%m-%d %H:%M:%S"))
                label = st.selectbox(
                    "Format",
                    labels,
                    key=f"cached_files_variant_{artifact_id}",
                ) if len(labels) > 1 else labels[0]
            variant = variants[label]
            direct_payload, file_name, mime = selected_download_payload(variant, compress=False)
            with actions:
                action_cols = st.columns(3, gap="small")
                action_cols[0].download_button(
                    "Download",
                    data=direct_payload,
                    file_name=file_name,
                    mime=mime,
                    key=f"cached_files_download_{artifact_id}_{label}",
                    on_click="ignore",
                    width="stretch",
                )
                if action_cols[1].button(
                    "Prepare GZ",
                    key=f"cached_files_gzip_{artifact_id}_{label}",
                    help="Compression runs only after this button is pressed.",
                    width="stretch",
                ):
                    with st.spinner("Compressing selected export..."):
                        gz_payload, gz_name, gz_mime = selected_download_payload(variant, compress=True)
                    st.session_state[GZIP_PREVIEW_KEY] = {
                        "artifact_id": artifact_id,
                        "label": label,
                        "payload": gz_payload,
                        "file_name": gz_name,
                        "mime": gz_mime,
                    }
                if action_cols[2].button(
                    "Remove",
                    key=f"cached_files_remove_{artifact_id}",
                    width="stretch",
                ):
                    remove_generated_artifact(artifact_id)
                    st.session_state.pop(GZIP_PREVIEW_KEY, None)
                    _rerun()
            prepared = st.session_state.get(GZIP_PREVIEW_KEY)
            if (
                isinstance(prepared, dict)
                and prepared.get("artifact_id") == artifact_id
                and prepared.get("label") == label
            ):
                st.download_button(
                    "Download prepared gzip",
                    data=prepared["payload"],
                    file_name=prepared["file_name"],
                    mime=prepared["mime"],
                    key=f"cached_files_download_gzip_{artifact_id}_{label}",
                    on_click="ignore",
                )


def _render_uploads() -> None:
    records = sorted(uploads(), key=lambda item: item.created_at, reverse=True)
    total = sum(item.size_bytes for item in records)
    st.caption(f"{len(records)} content-addressed upload(s) - {memory_label(total)}")
    shown, page, pages = _page_slice(records, "uploads")
    if not shown:
        st.info("No uploaded sources are retained.")
        return
    st.caption(f"Page {page} of {pages}")
    for record in shown:
        with st.container(border=True):
            columns = st.columns((0.55, 0.17, 0.28), gap="small")
            columns[0].write(record.name)
            columns[0].caption(f"SHA-256 {record.digest[:16]}...")
            columns[1].write(memory_label(record.size_bytes))
            actions = columns[2].columns(2, gap="small")
            actions[0].download_button(
                "Download",
                data=record.payload,
                file_name=record.name,
                mime=record.mime,
                key=f"cached_files_upload_download_{record.digest}",
                on_click="ignore",
                width="stretch",
            )
            if actions[1].button(
                "Remove",
                key=f"cached_files_upload_remove_{record.digest}",
                width="stretch",
            ):
                remove_upload(record.digest)
                _rerun()


def _render_datasets() -> None:
    records = [item for item in datasets() if isinstance(item, DatasetRecord)]
    records.sort(key=lambda item: item.created_at, reverse=True)
    total = sum(record.nbytes for record in records)
    st.caption(f"{len(records)} canonical dataset(s) - {memory_label(total)}")
    shown, page, pages = _page_slice(records, "datasets")
    if not shown:
        st.info("No canonical datasets are prepared. Use Data Management to create one.")
        return
    st.caption(f"Page {page} of {pages}")
    for record in shown:
        with st.container(border=True):
            columns = st.columns((0.55, 0.17, 0.28), gap="small")
            columns[0].write(dataset_display_name(record))
            columns[0].caption(f"Dataset {record.dataset_id[:16]}... - source {record.source_digest[:12]}...")
            columns[1].write(memory_label(record.nbytes))
            actions = columns[2].columns(2, gap="small")
            if actions[0].button(
                "Open plot",
                key=f"cached_files_plot_{record.dataset_id}",
                width="stretch",
            ):
                st.session_state["plotting_profile_source_mode"] = "Prepared data or result"
                st.session_state["plotting_data_view"] = f"dataset:{record.dataset_id}"
                st.session_state["dnascape_active_page"] = "plotting"
                _rerun()
            if actions[1].button(
                "Remove",
                key=f"cached_files_dataset_remove_{record.dataset_id}",
                width="stretch",
            ):
                remove_dataset(record.dataset_id)
                _rerun()


def _render_workflows() -> None:
    results = workflow_results()
    if not results:
        st.info("No shared mapping or simulation results are registered.")
        return
    for workflow, result in sorted(results.items()):
        with st.container(border=True):
            columns = st.columns((0.55, 0.2, 0.25), gap="small")
            columns[0].write(workflow.title())
            columns[0].caption(
                f"Input signature {str(result.input_signature)[:18]}... - "
                f"created {_display_datetime(result.created_at):%Y-%m-%d %H:%M:%S}"
            )
            columns[1].write(memory_label(_payload_nbytes(result.payload)))
            if columns[2].button(
                "Remove result",
                key=f"cached_files_workflow_remove_{workflow}",
                width="stretch",
            ):
                clear_workflow_result(workflow)
                _rerun()


def _render_retained_session_data() -> None:
    """Expose canonical inputs/results that remain usable across page changes."""

    st.markdown("### Retained session data")
    st.caption(
        "Uploads, prepared datasets, and completed workflow results are retained "
        "independently from generated download files."
    )
    upload_tab, dataset_tab, workflow_tab = st.tabs(
        ("Uploads", "Prepared datasets", "Workflow results")
    )
    with upload_tab:
        _render_uploads()
    with dataset_tab:
        _render_datasets()
    with workflow_tab:
        _render_workflows()


def render_cached_files_workspace() -> None:
    discarded_artifacts = _sanitize_generated_artifact_state()
    with st.container(border=True, key="dnascape-cache-card"):
        st.markdown(
            '<div class="cache-settings-title">Cache settings</div>',
            unsafe_allow_html=True,
        )

        artifacts = generated_artifacts()
        has_artifacts = bool(artifacts)
        sort_key_name = persistent_widget_key(PAGE_NAMESPACE, "generated_sort")
        sort_desc_name = persistent_widget_key(PAGE_NAMESPACE, "generated_descending")
        retained_sort_key = st.session_state.get(sort_key_name)
        if not isinstance(retained_sort_key, str) or retained_sort_key not in {"date", "size"}:
            st.session_state[sort_key_name] = "date"
        if not isinstance(st.session_state.get(sort_desc_name), bool):
            st.session_state[sort_desc_name] = True
        sort_key = str(st.session_state[sort_key_name])
        descending = bool(st.session_state[sort_desc_name])
        date_label = f"Date {'↓' if descending else '↑'}" if sort_key == "date" else "Date"
        size_label = f"Size {'↓' if descending else '↑'}" if sort_key == "size" else "Size"

        controls = st.columns((0.1, 0.22, 0.18, 0.2, 0.3), gap="small")
        with controls[0]:
            current_limit = generated_cache_limit()
            limit_key = persistent_widget_key(PAGE_NAMESPACE, "limit")
            if not isinstance(st.session_state.get(limit_key), int):
                st.session_state[limit_key] = current_limit
            limit = st.number_input(
                "Max. files",
                min_value=1,
                max_value=MAX_GENERATED_CACHE_LIMIT,
                step=1,
                key=limit_key,
                help="Older generated files are automatically removed after this count.",
            )
            if int(limit) != current_limit:
                set_generated_cache_limit(limit)

        with controls[1]:
            st.markdown('<div class="cache-control-offset"></div>', unsafe_allow_html=True)
            toggle_cols = st.columns((0.55, 0.45), gap="small")
            with toggle_cols[0]:
                keep_raw_key = persistent_widget_key(PAGE_NAMESPACE, "keep_raw")
                if not isinstance(st.session_state.get(keep_raw_key), bool):
                    st.session_state[keep_raw_key] = generated_cache_keep_raw()
                keep_raw = st.checkbox(
                    "Raw output",
                    key=keep_raw_key,
                    help="Store newly generated mapping and simulation raw outputs.",
                )
                set_generated_cache_keep_raw(keep_raw)
            with toggle_cols[1]:
                keep_plots_key = persistent_widget_key(PAGE_NAMESPACE, "keep_plots")
                if not isinstance(st.session_state.get(keep_plots_key), bool):
                    st.session_state[keep_plots_key] = generated_cache_keep_plots()
                keep_plots = st.checkbox(
                    "Plots",
                    key=keep_plots_key,
                    help="Store newly generated preview plot exports.",
                )
                set_generated_cache_keep_plots(keep_plots)

        with controls[2]:
            st.markdown('<div class="cache-control-offset"></div>', unsafe_allow_html=True)
            sort_cols = st.columns(2, gap="small")
            if sort_cols[0].button(
                date_label,
                type="primary" if sort_key == "date" else "secondary",
                disabled=not has_artifacts,
                width="stretch",
                key="cached_files_sort_date",
                help="Sort by creation date. Click again to reverse direction.",
            ):
                if sort_key == "date":
                    st.session_state[sort_desc_name] = not descending
                else:
                    st.session_state[sort_key_name] = "date"
                    st.session_state[sort_desc_name] = True
                _rerun()
            if sort_cols[1].button(
                size_label,
                type="primary" if sort_key == "size" else "secondary",
                disabled=not has_artifacts,
                width="stretch",
                key="cached_files_sort_size",
                help="Sort by size. Click again to reverse direction.",
            ):
                if sort_key == "size":
                    st.session_state[sort_desc_name] = not descending
                else:
                    st.session_state[sort_key_name] = "size"
                    st.session_state[sort_desc_name] = True
                _rerun()

        with controls[3]:
            st.markdown('<div class="cache-control-offset"></div>', unsafe_allow_html=True)
            if st.button(
                "Clear generated cache",
                icon=":material/delete:",
                type="secondary",
                disabled=not has_artifacts,
                width="stretch",
                key="cached_files_clear_generated",
            ):
                clear_generated_artifacts()
                st.session_state.pop(GZIP_PREVIEW_KEY, None)
                _rerun()

        artifacts = sorted_generated_artifacts(artifacts, sort_key, descending)
        total = sum(generated_artifact_size(item) for item in artifacts)
        direction = "descending" if descending else "ascending"
        with controls[4]:
            st.markdown('<div class="cache-control-offset"></div>', unsafe_allow_html=True)
            st.caption(
                f"{len(artifacts)} generated file(s) | {artifact_size_label(total)} "
                f"| {sort_key} {direction}"
            )

        if discarded_artifacts:
            st.warning(
                f"Ignored {discarded_artifacts} damaged cache "
                f"entr{'y' if discarded_artifacts == 1 else 'ies'}."
            )

        if not artifacts:
            st.markdown(
                """
                <p class="small-muted">
                No generated files. Create an export in Data Management, Mapping,
                Simulation, or Plotting to add it here.
                </p>
                """,
                unsafe_allow_html=True,
            )

        for row_index, artifact in enumerate(artifacts):
            artifact_id = str(artifact.get("id", ""))
            variants = artifact_download_variants(artifact)
            if not variants:
                continue
            variant_labels = list(variants)
            default_variant = variant_labels.index("CSV") if "CSV" in variant_labels else 0
            file_name = str(artifact.get("file_name") or "dnascape-output.bin")
            source = str(artifact.get("source") or "DNAscape")
            category = str(artifact.get("category") or "Generated output")
            artifact_type = str(artifact.get("artifact_type") or "file").casefold()
            summary = str(artifact.get("summary") or "")
            description = str(artifact.get("description") or "")
            created = _display_datetime(artifact.get("created_at", 0.0))
            can_compress = artifact_type == "raw" or category == "Raw output"

            with st.container(border=True):
                row = st.columns((0.43, 0.14, 0.1, 0.14, 0.055, 0.09), gap="small")
                with row[0]:
                    st.markdown(f"**{file_name}**")
                    if summary:
                        st.markdown(
                            f'<div class="cache-artifact-summary">{html.escape(summary)}</div>',
                            unsafe_allow_html=True,
                        )
                    metadata = f"{source} | {category}"
                    if description:
                        metadata = f"{metadata} | {description}"
                    st.caption(metadata)
                with row[1]:
                    st.caption("Created")
                    st.write(created.strftime("%Y-%m-%d %H:%M:%S"))
                with row[2]:
                    st.caption("Size")
                    st.write(artifact_size_label(generated_artifact_size(artifact)))
                with row[3]:
                    if len(variant_labels) > 1:
                        format_key = f"cached_format_{artifact_id}"
                        if st.session_state.get(format_key) not in variant_labels:
                            st.session_state[format_key] = variant_labels[default_variant]
                        selected_label = st.selectbox(
                            "Format",
                            variant_labels,
                            key=format_key,
                        )
                    else:
                        selected_label = variant_labels[0]
                        st.caption("Format")
                        st.write(selected_label)
                with row[4]:
                    st.markdown('<div class="cache-row-offset"></div>', unsafe_allow_html=True)
                    compress = False
                    if can_compress:
                        gzip_key = f"cached_gz_{artifact_id}"
                        if not isinstance(st.session_state.get(gzip_key), bool):
                            st.session_state[gzip_key] = False
                        compress = st.checkbox(
                            "GZ",
                            key=gzip_key,
                        )

                variant = variants[selected_label]
                if compress:
                    _direct_payload, direct_name, _direct_mime = selected_download_payload(
                        variant,
                        compress=False,
                    )
                    payload = partial(_compressed_download_bytes, variant)
                    download_name = f"{direct_name}.gz"
                    mime = "application/gzip"
                else:
                    payload, download_name, mime = selected_download_payload(
                        variant,
                        compress=False,
                    )

                with row[5]:
                    st.markdown('<div class="cache-row-offset"></div>', unsafe_allow_html=True)
                    with st.container(key=f"cached-file-actions-{row_index}"):
                        action_cols = st.columns(2, gap="small")
                        with action_cols[0]:
                            with st.container(key=f"cached-file-icon-download-{row_index}"):
                                st.download_button(
                                    "Download",
                                    data=payload,
                                    file_name=download_name,
                                    mime=mime,
                                    key=(
                                        f"cached_download_{artifact_id}_{selected_label}_"
                                        f"{int(compress)}"
                                    ),
                                    help="Download",
                                    icon=":material/download:",
                                    on_click="ignore",
                                    width="content",
                                )
                        with action_cols[1]:
                            with st.container(key=f"cached-file-icon-remove-{row_index}"):
                                if st.button(
                                    "Remove",
                                    key=f"cached_remove_{artifact_id}",
                                    help="Remove",
                                    icon=":material/delete:",
                                    width="content",
                                ):
                                    remove_generated_artifact(artifact_id)
                                    st.session_state.pop(GZIP_PREVIEW_KEY, None)
                                    _rerun()

    _render_retained_session_data()


def render_page() -> None:
    """Render Cached Files inside the persistent DNAscape shell."""

    render_page_intro(
        "Cached Files",
        "Manage generated files",
        "Review and download data and figures saved during this DNAscape session.",
    )
    render_standard_content_marker("cached-files-page-marker")
    st.markdown(
        """
        <style>
        body:has(.cached-files-page-marker) .cache-settings-title {
          color: var(--dnascape-accent); font-size: 0.88rem; font-weight: 700;
          margin: 0 0 0.35rem 0; text-transform: uppercase;
        }
        body:has(.cached-files-page-marker) .cache-control-offset { height: 1.55rem; }
        body:has(.cached-files-page-marker) .cache-row-offset { height: 1.52rem; }
        body:has(.cached-files-page-marker) .cache-artifact-summary {
          color: var(--dnascape-muted); font-size: 0.88rem; line-height: 1.25;
          margin: -0.12rem 0 0.3rem 0;
        }
        body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-"] button {
          align-items: center; display: inline-flex !important; height: 2.3rem;
          justify-content: center; min-height: 2.3rem; min-width: 2.3rem;
          padding: 0 !important; width: 2.3rem;
        }
        body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-"] button p {
          display: none !important;
        }
        body:has(.cached-files-page-marker) [class*="st-key-cached-file-actions-"]
        [data-testid="stHorizontalBlock"] { gap: 0.38rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="dnascape-standard-content"):
        render_cached_files_workspace()
