"""Cached Files page for the DNAscape Streamlit prototype."""

from datetime import datetime
import gzip
import html
import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_shell import (
    MAX_GENERATED_CACHE_LIMIT,
    artifact_size_label,
    clear_generated_artifacts,
    configure_page,
    generated_artifact_size,
    generated_artifacts,
    generated_cache_limit,
    generated_cache_keep_plots,
    generated_cache_keep_raw,
    remove_generated_artifact,
    render_page_intro,
    render_sidebar,
    render_standard_content_marker,
    set_generated_cache_limit,
    set_generated_cache_keep_plots,
    set_generated_cache_keep_raw,
)


CACHE_SORT_KEY = "cached_files_sort_key"
CACHE_SORT_DESC_KEY = "cached_files_sort_desc"
DEFAULT_CACHE_SORT = "date"


def artifact_download_variants(
    artifact: dict[str, object],
) -> dict[str, dict[str, object]]:
    variants = artifact.get("variants")
    if isinstance(variants, dict) and variants:
        return {
            str(label): variant
            for label, variant in variants.items()
            if isinstance(variant, dict) and variant.get("payload") is not None
        }

    payload = artifact.get("payload", b"")
    return {
        "File": {
            "file_name": artifact.get("file_name", "dnascape-output.bin"),
            "mime": artifact.get("mime", "application/octet-stream"),
            "payload": payload,
        }
    }


def selected_download_payload(
    variant: dict[str, object],
    *,
    compress: bool,
) -> tuple[bytes, str, str]:
    payload = bytes(variant.get("payload", b""))
    file_name = str(variant.get("file_name") or "dnascape-output.bin")
    mime = str(variant.get("mime") or "application/octet-stream")
    if not compress:
        return payload, file_name, mime
    return gzip.compress(payload), f"{file_name}.gz", "application/gzip"


def cache_sort_state() -> tuple[str, bool]:
    sort_key = str(st.session_state.get(CACHE_SORT_KEY, DEFAULT_CACHE_SORT))
    if sort_key not in {"date", "size"}:
        sort_key = DEFAULT_CACHE_SORT
    descending = bool(st.session_state.get(CACHE_SORT_DESC_KEY, True))
    return sort_key, descending


def set_cache_sort(sort_key: str) -> None:
    current_sort, current_descending = cache_sort_state()
    if sort_key == current_sort:
        st.session_state[CACHE_SORT_DESC_KEY] = not current_descending
    else:
        st.session_state[CACHE_SORT_KEY] = sort_key
        st.session_state[CACHE_SORT_DESC_KEY] = True


def sorted_generated_artifacts(
    artifacts: list[dict[str, object]],
    sort_key: str,
    descending: bool,
) -> list[dict[str, object]]:
    if sort_key == "size":
        return sorted(
            artifacts,
            key=lambda artifact: (
                generated_artifact_size(artifact),
                float(artifact.get("created_at", 0.0)),
            ),
            reverse=descending,
        )

    return sorted(
        artifacts,
        key=lambda artifact: (
            float(artifact.get("created_at", 0.0)),
            generated_artifact_size(artifact),
        ),
        reverse=descending,
    )


def rerun_cached_files_fragment() -> None:
    """Rerun only the Cached Files workspace fragment when Streamlit allows it."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx(suppress_warning=True)
    if ctx is not None and ctx.fragment_ids_this_run:
        st.rerun(scope="fragment")
    st.rerun()


configure_page("DNAscape | Cached Files")
render_sidebar("Cached Files")

render_page_intro(
    "Cached Files",
    "Manage reusable app artifacts",
    "Review generated data, intermediate tables, plots, simulation outputs, "
    "and other files cached during a DNAscape session.",
)

render_standard_content_marker("cached-files-page-marker")

st.markdown(
    """
    <style>
    body:has(.cached-files-page-marker) .cache-settings-title {
      color: var(--dnascape-accent);
      font-size: 0.88rem;
      font-weight: 700;
      letter-spacing: 0;
      margin: 0 0 0.35rem 0;
      text-transform: uppercase;
    }

    body:has(.cached-files-page-marker) .cache-control-offset {
      height: 1.55rem;
    }

    body:has(.cached-files-page-marker) .cache-row-offset {
      height: 1.52rem;
    }

    body:has(.cached-files-page-marker) .cache-artifact-summary {
      color: var(--dnascape-muted);
      font-size: 0.88rem;
      line-height: 1.25;
      margin: -0.12rem 0 0.3rem 0;
    }

    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-download-"] button,
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-remove-"] button {
      align-items: center;
      display: inline-flex !important;
      gap: 0 !important;
      height: 2.3rem;
      justify-content: center;
      min-height: 2.3rem;
      min-width: 2.3rem;
      padding: 0 !important;
      position: relative !important;
      width: 2.3rem;
    }

    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-download-"] button > div,
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-remove-"] button > div,
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-download-"] button span,
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-remove-"] button span {
      align-items: center !important;
      display: inline-flex !important;
      justify-content: center !important;
      margin: 0 !important;
    }

    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-download-"] button svg,
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-remove-"] button svg,
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-download-"] button span[class*="material"],
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-remove-"] button span[class*="material"],
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-download-"] button [data-testid="stIconMaterial"],
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-remove-"] button [data-testid="stIconMaterial"] {
      left: 50% !important;
      line-height: 1 !important;
      margin: 0 !important;
      position: absolute !important;
      top: 50% !important;
      transform: translate(-50%, -50%) !important;
    }

    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-download-"] button p,
    body:has(.cached-files-page-marker) [class*="st-key-cached-file-icon-remove-"] button p {
      display: none !important;
      margin: 0 !important;
    }

    body:has(.cached-files-page-marker) [class*="st-key-cached-file-actions-"] [data-testid="stHorizontalBlock"] {
      gap: 0.38rem !important;
      justify-content: flex-start !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.fragment
def render_cached_files_workspace() -> None:
    with st.container(key="dnascape-standard-content"):
        with st.container(border=True, key="dnascape-cache-card"):
            st.markdown(
                '<div class="cache-settings-title">Cache settings</div>',
                unsafe_allow_html=True,
            )

            current_limit = generated_cache_limit()
            keep_raw = generated_cache_keep_raw()
            keep_plots = generated_cache_keep_plots()
            sort_key, sort_descending = cache_sort_state()
            cached_artifacts = generated_artifacts()
            has_artifacts = bool(cached_artifacts)
            date_label = "Date"
            size_label_sort = "Size"
            if sort_key == "date":
                date_label = f"Date {'↓' if sort_descending else '↑'}"
            elif sort_key == "size":
                size_label_sort = f"Size {'↓' if sort_descending else '↑'}"

            controls = st.columns([0.1, 0.22, 0.18, 0.2, 0.3], gap="small")
            with controls[0]:
                cache_limit = st.number_input(
                    "Max. files",
                    min_value=1,
                    max_value=MAX_GENERATED_CACHE_LIMIT,
                    value=current_limit,
                    step=1,
                    help="Older generated files are automatically removed after this count.",
                )
                if int(cache_limit) != current_limit:
                    set_generated_cache_limit(cache_limit)
                    rerun_cached_files_fragment()

            with controls[1]:
                st.markdown('<div class="cache-control-offset"></div>', unsafe_allow_html=True)
                toggle_cols = st.columns([0.55, 0.45], gap="small")
                with toggle_cols[0]:
                    keep_raw_next = st.checkbox(
                        "Raw output",
                        value=keep_raw,
                        help="Store newly generated mapping and simulation raw outputs.",
                    )
                    if keep_raw_next != keep_raw:
                        set_generated_cache_keep_raw(keep_raw_next)
                        rerun_cached_files_fragment()
                with toggle_cols[1]:
                    keep_plots_next = st.checkbox(
                        "Plots",
                        value=keep_plots,
                        help="Store newly generated preview plot exports.",
                    )
                    if keep_plots_next != keep_plots:
                        set_generated_cache_keep_plots(keep_plots_next)
                        rerun_cached_files_fragment()

            with controls[2]:
                st.markdown('<div class="cache-control-offset"></div>', unsafe_allow_html=True)
                sort_cols = st.columns([1, 1], gap="small")
                with sort_cols[0]:
                    if st.button(
                        date_label,
                        type="primary" if sort_key == "date" else "secondary",
                        width="stretch",
                        disabled=not has_artifacts,
                        help=(
                            "Sort by creation date. Click again to reverse direction."
                        ),
                    ):
                        set_cache_sort("date")
                        rerun_cached_files_fragment()
                with sort_cols[1]:
                    if st.button(
                        size_label_sort,
                        type="primary" if sort_key == "size" else "secondary",
                        width="stretch",
                        disabled=not has_artifacts,
                        help="Sort by size. Click again to reverse direction.",
                    ):
                        set_cache_sort("size")
                        rerun_cached_files_fragment()

            with controls[3]:
                st.markdown('<div class="cache-control-offset"></div>', unsafe_allow_html=True)
                if st.button(
                    "Clear generated cache",
                    icon=":material/delete:",
                    type="secondary",
                    width="stretch",
                    disabled=not has_artifacts,
                ):
                    clear_generated_artifacts()
                    rerun_cached_files_fragment()

            artifacts = sorted_generated_artifacts(
                cached_artifacts,
                sort_key,
                sort_descending,
            )
            total_size = sum(generated_artifact_size(artifact) for artifact in artifacts)
            sort_direction = "descending" if sort_descending else "ascending"
            with controls[4]:
                st.markdown('<div class="cache-control-offset"></div>', unsafe_allow_html=True)
                st.caption(
                    f"{len(artifacts)} generated file(s) | {artifact_size_label(total_size)} "
                    f"| {sort_key} {sort_direction}"
                )

            if not artifacts:
                st.markdown(
                    """
                    <p class="small-muted">
                    No generated files are cached yet. Run a conversion, mapping,
                    simulation, or plot to populate this list.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                for row_index, artifact in enumerate(artifacts):
                    artifact_id = str(artifact.get("id", ""))
                    file_name = str(artifact.get("file_name", "dnascape-output.bin"))
                    source = str(artifact.get("source", "DNAscape"))
                    category = str(artifact.get("category", "Generated output"))
                    artifact_type = str(artifact.get("artifact_type", "file")).casefold()
                    summary = str(artifact.get("summary", ""))
                    description = str(artifact.get("description", ""))
                    created_at = datetime.fromtimestamp(
                        float(artifact.get("created_at", 0.0))
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    variants = artifact_download_variants(artifact)
                    variant_labels = list(variants)
                    default_variant_index = (
                        variant_labels.index("CSV") if "CSV" in variant_labels else 0
                    )
                    can_compress = artifact_type == "raw" or category == "Raw output"
                    size_label = artifact_size_label(generated_artifact_size(artifact))

                    with st.container(border=True):
                        row = st.columns(
                            [0.43, 0.14, 0.1, 0.14, 0.055, 0.09],
                            gap="small",
                        )
                        with row[0]:
                            st.markdown(f"**{file_name}**")
                            if summary:
                                st.markdown(
                                    f'<div class="cache-artifact-summary">{html.escape(summary)}</div>',
                                    unsafe_allow_html=True,
                                )
                            meta = f"{source} | {category}"
                            if description:
                                meta = f"{meta} | {description}"
                            st.caption(meta)
                        with row[1]:
                            st.caption("Created")
                            st.write(created_at)
                        with row[2]:
                            st.caption("Size")
                            st.write(size_label)
                        with row[3]:
                            if len(variant_labels) > 1:
                                selected_variant_label = st.selectbox(
                                    "Format",
                                    variant_labels,
                                    index=default_variant_index,
                                    key=f"cached_format_{artifact_id}",
                                )
                            else:
                                selected_variant_label = variant_labels[0]
                                st.caption("Format")
                                st.write(selected_variant_label)
                        with row[4]:
                            st.markdown('<div class="cache-row-offset"></div>', unsafe_allow_html=True)
                            compress = False
                            if can_compress:
                                compress = st.checkbox(
                                    "GZ",
                                    value=False,
                                    key=f"cached_gz_{artifact_id}",
                                )
                            else:
                                st.markdown(
                                    "<div style='height: 1.75rem'></div>",
                                    unsafe_allow_html=True,
                                )
                        with row[5]:
                            st.markdown('<div class="cache-row-offset"></div>', unsafe_allow_html=True)
                            selected_variant = variants[selected_variant_label]
                            payload, download_file_name, mime = selected_download_payload(
                                selected_variant,
                                compress=compress,
                            )
                            with st.container(key=f"cached-file-actions-{row_index}"):
                                action_cols = st.columns([1, 1], gap="small")
                                with action_cols[0]:
                                    with st.container(key=f"cached-file-icon-download-{row_index}"):
                                        st.download_button(
                                            "Download",
                                            data=payload,
                                            file_name=download_file_name,
                                            mime=mime,
                                            key=f"cached_download_{artifact_id}",
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
                                            rerun_cached_files_fragment()


render_cached_files_workspace()
