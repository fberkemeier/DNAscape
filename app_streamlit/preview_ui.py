"""Compact image previews with the DNAscape hover action overlay."""

from __future__ import annotations

import base64
import html
import re
from collections.abc import Mapping
from typing import Any, Callable

import streamlit as st

_DOWNLOAD_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <path d="M12 3v10.2l3.6-3.6 1.4 1.4-6 6-6-6 1.4-1.4 3.6 3.6V3h2zm-7 14h14v2H5v-2z" fill="currentColor" />
</svg>
"""
_CLEAR_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <path d="M9 3h6l1 1h4v2H4V4h4l1-1Zm-3 4h12v12a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V7Zm3 3v8h2v-8H9Zm4 0v8h2v-8h-2Z" fill="currentColor" />
</svg>
"""
_EXPAND_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <path d="M5 5h6v2H8.4l3.3 3.3-1.4 1.4L7 8.4V11H5V5zm8 0h6v6h-2V8.4l-3.3 3.3-1.4-1.4L15.6 7H13V5zM7 15.6l3.3-3.3 1.4 1.4L8.4 17H11v2H5v-6h2v2.6zm9.6 1.4-3.3-3.3 1.4-1.4 3.3 3.3V13h2v6h-6v-2h2.6z" fill="currentColor" />
</svg>
"""


def render_image_preview(
    payload: bytes,
    *,
    file_name: str,
    shell_key: str,
    clear_key: str,
    on_clear: Callable[..., Any],
    clear_args: tuple[Any, ...] = (),
    mime: str = "image/png",
    alt: str = "DNAscape plot preview",
    downloads: Mapping[str, tuple[bytes, str, str]] | None = None,
) -> None:
    """Render the repository-style preview with download, clear, and expand actions."""

    try:
        payload_bytes = bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("Preview payload must be bytes-like.") from exc
    if not payload_bytes:
        raise ValueError("Preview payload cannot be empty.")
    if not callable(on_clear):
        raise TypeError("Preview clear action must be callable.")

    encoded = base64.b64encode(payload_bytes).decode("ascii")
    safe_alt = html.escape(str(alt), quote=True)
    toggle_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", f"{shell_key}-{file_name}-expand")
    variants = (
        {"png": (payload_bytes, mime, file_name)}
        if downloads is None
        else downloads
    )
    download_links = []
    for label, variant in variants.items():
        if not isinstance(variant, tuple) or len(variant) != 3:
            raise ValueError(
                "Each preview download must contain payload, MIME type, and file name."
            )
        variant_payload, variant_mime, variant_name = variant
        try:
            variant_bytes = bytes(variant_payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Download payload {label!r} must be bytes-like.") from exc
        if not variant_bytes:
            raise ValueError(f"Download payload {label!r} cannot be empty.")
        variant_encoded = (
            encoded
            if variant_payload is payload or variant_bytes is payload_bytes
            else base64.b64encode(variant_bytes).decode("ascii")
        )
        download_links.append(
            '<a href="data:{mime};base64,{payload}" download="{name}">{label}</a>'.format(
                mime=html.escape(str(variant_mime), quote=True),
                payload=variant_encoded,
                name=html.escape(str(variant_name), quote=True),
                label=html.escape(str(label).lower()),
            )
        )
    download_menu = ""
    if download_links:
        download_menu = f"""
                <div class="plot-download-menu">
                  <button class="plot-preview-action" type="button" aria-label="Download preview"
                          aria-haspopup="menu" title="Download">
                    {_DOWNLOAD_ICON}
                  </button>
                  <div class="plot-download-menu-panel" role="menu">
                    {''.join(download_links)}
                  </div>
                </div>
        """

    with st.container(key=shell_key):
        st.button(
            "Clear preview",
            key=clear_key,
            icon=":material/delete:",
            on_click=on_clear,
            args=clear_args,
        )
        # Streamlit's Markdown parser can expose fragments of very large data
        # URIs as text. ``st.html`` hands the same repository-style markup to
        # the HTML sanitizer directly, so the preview and download links remain
        # intact for PNG, SVG, and PDF payloads.
        st.html(
            f"""
            <div class="plot-preview-frame">
              <input class="plot-preview-expand-toggle" id="{toggle_id}" type="checkbox" aria-hidden="true" />
              <div class="plot-preview-actions" aria-label="Plot preview actions">
                {download_menu}
                <span class="plot-preview-action plot-preview-clear-placeholder" aria-hidden="true">
                  {_CLEAR_ICON}
                </span>
                <label class="plot-preview-action plot-preview-expand-button" for="{toggle_id}"
                       aria-label="Expand preview" title="Expand preview">
                  {_EXPAND_ICON}
                </label>
              </div>
              <img class="plot-preview-image" src="data:{html.escape(mime, quote=True)};base64,{encoded}"
                   alt="{safe_alt}" />
            </div>
            """
        )
