"""Data Management page for the DNAscape Streamlit prototype."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_shell import configure_page, render_page_intro, render_sidebar


configure_page("DNAscape | Data Management")
render_sidebar("Data Management")

render_page_intro(
    "Data Management",
    "Convert genomic data files",
    "Upload a supported file and convert it between DNAscape-compatible formats.",
)

st.markdown(
    """
    <span class="data-management-page-marker" aria-hidden="true"></span>
    <style>
      .data-management-page-marker {
        display: none !important;
      }

      body:has(.data-management-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
        gap: 0.72rem !important;
      }

      body:has(.data-management-page-marker) [data-testid="stHorizontalBlock"],
      body:has(.data-management-page-marker) [data-testid="column"] {
        min-width: 0 !important;
      }

      .data-management-panel-title {
        color: var(--dnascape-text);
        font-size: 1.05rem;
        font-weight: 650;
        line-height: 1.35;
        margin: 0 0 0.55rem;
      }

      .data-management-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.2rem 0 0.35rem;
      }

      .data-management-chip {
        background: color-mix(in srgb, var(--dnascape-text) 5%, var(--dnascape-panel));
        border: 1px solid var(--dnascape-line);
        border-radius: 999px;
        color: var(--dnascape-muted);
        font-size: 0.82rem;
        line-height: 1.2;
        padding: 0.34rem 0.58rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


TARGET_FORMATS = {
    "CSV": "csv",
    "TSV": "tsv",
    "TXT": "txt",
    "NPV": "npv",
    "NPY": "npy",
    "NPZ": "npz",
}
TARGET_MIME = {
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "txt": "text/plain",
    "npy": "application/octet-stream",
    "npv": "application/octet-stream",
    "npz": "application/octet-stream",
}
DATA_FORM_LABELS = {
    "Auto": "auto",
    "Table": "table",
    "Space": "space",
    "Time": "time",
    "Space-time": "spacetime",
}
FORMAT_LABELS = {
    "csv": "CSV",
    "tsv": "TSV",
    "txt": "TXT",
    "npv": "NPV",
    "npy": "NPY",
    "npz": "NPZ",
}
UPLOAD_TYPES = [
    "csv",
    "tsv",
    "txt",
    "npv",
    "npy",
    "npz",
    "gz",
    "bw",
    "bigwig",
    "bedgraph",
    "bdg",
]
FORMAT_EXAMPLES = (
    "CSV, TSV, TXT, NPV, NPY, NPZ, GZ-compressed CSV/TSV/TXT, "
    "bigWig, and bedGraph files."
)
LAST_CONVERSION_KEY = "data_management_last_conversion"


def upload_token(uploaded_file) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"


def upload_suffix(name: str) -> str:
    suffixes = Path(name).suffixes
    if not suffixes:
        return ".dat"
    if suffixes[-1].lower() == ".gz" and len(suffixes) >= 2:
        return "".join(suffixes[-2:])
    return suffixes[-1]


def upload_stem(name: str) -> str:
    output = Path(name).name
    for suffix in reversed(Path(name).suffixes):
        if suffix.lower() in {".csv", ".tsv", ".txt", ".npv", ".npy", ".npz", ".gz"}:
            output = output[: -len(suffix)]
    return output or "dnascape_data"


def detect_source_format(name: str) -> str | None:
    suffixes = [suffix.lower() for suffix in Path(name).suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    if not suffixes:
        return None
    suffix = suffixes[-1].lstrip(".")
    return suffix if suffix in {"csv", "tsv", "txt", "npv", "npy", "npz"} else None


def safe_stem(*parts: object) -> str:
    raw_name = "_".join(str(part) for part in parts if part)
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in raw_name)


def convert_with_rdata(
    uploaded_file,
    *,
    source_format: str,
    target_format: str,
    compress_output: bool,
    data_form: str,
    column_name: str | None,
    array_key: str | None,
    decimal_places: int | None,
    value_name: str,
) -> dict[str, object]:
    from dnascape.io import rdata

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_path = tmp_path / f"input{upload_suffix(uploaded_file.name)}"
        file_stem = safe_stem(upload_stem(uploaded_file.name))
        output_extension = f"{target_format}.gz" if compress_output else target_format
        output_path = tmp_path / f"{file_stem}.{output_extension}"
        source_path.write_bytes(uploaded_file.getvalue())

        column_kwargs = {"chromosome": column_name} if column_name else {}
        result = rdata(
            source_path,
            target=output_path,
            source_format=source_format,
            data_form=data_form,
            array_key=array_key,
            decimal_places=decimal_places,
            value_name=value_name,
            return_data=True,
            **column_kwargs,
        )
        payload = output_path.read_bytes()

    return {
        "payload": payload,
        "file_stem": file_stem,
        "extension": output_extension,
        "mime": "application/gzip" if compress_output else TARGET_MIME[target_format],
        "compressed": compress_output,
        "data_form": result["data_form"],
        "source_format": result["source_format"],
        "target_format": result["target_format"],
        "values_shape": tuple(np.asarray(result["values"]).shape),
    }


left, right = st.columns([0.35, 0.65], gap="medium")

uploaded_file = None
source_format = None
source_error = None
convert_clicked = False

with left:
    with st.container(border=True):
        st.markdown('<div class="data-management-panel-title">Input</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload",
            type=UPLOAD_TYPES,
            help=f"Examples: {FORMAT_EXAMPLES}",
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            source_format = detect_source_format(uploaded_file.name)
            if source_format is None:
                source_error = (
                    "Could not detect a currently supported rdata format from the file name. "
                    "Supported conversion formats are CSV, TSV, TXT, NPV, NPY, and NPZ. "
                    "bigWig and bedGraph uploads are accepted for upcoming workflows."
                )

            if source_error:
                st.error(source_error)
            else:
                chips = [
                    f"Input: {FORMAT_LABELS[source_format]}",
                    f"{uploaded_file.size / 1024:,.1f} KB",
                ]
                if uploaded_file.name.lower().endswith(".gz"):
                    chips.append("GZ compressed")

                st.markdown(
                    "<div class='data-management-chip-row'>"
                    + "".join(f"<span class='data-management-chip'>{chip}</span>" for chip in chips)
                    + "</div>",
                    unsafe_allow_html=True,
                )

    with st.container():
        convert_button_slot = st.empty()

with right:
    with st.container(border=True):
        st.markdown(
            '<div class="data-management-panel-title">Conversion</div>',
            unsafe_allow_html=True,
        )

        conversion_ready = (
            uploaded_file is not None
            and source_error is None
            and source_format is not None
        )
        conversion_spec = None
        if not conversion_ready:
            st.markdown(
                """
                <p class="small-muted">
                Choose an input file and output format to define an rdata conversion.
                </p>
                """,
                unsafe_allow_html=True,
            )
        else:
            route_cols = st.columns([0.42, 0.42, 0.16], gap="small")
            with route_cols[0]:
                st.selectbox(
                    "From",
                    [FORMAT_LABELS[source_format]],
                    index=0,
                    disabled=True,
                )
            with route_cols[1]:
                target_label = st.selectbox(
                    "To",
                    list(TARGET_FORMATS),
                    index=list(TARGET_FORMATS).index("TSV"),
                )
            with route_cols[2]:
                compress_output = st.checkbox(
                    "GZ",
                    value=False,
                    key="data_management_convert_gz",
                )

            options_cols = st.columns(2, gap="small")
            with options_cols[0]:
                data_form_label = st.selectbox(
                    "Data form",
                    list(DATA_FORM_LABELS),
                    index=0,
                    help="rdata shape convention for arrays and vectors.",
                )
            with options_cols[1]:
                value_name = st.text_input(
                    "Column label",
                    value="value",
                    help="Column label used when writing vector data to CSV/TSV.",
                )

            column_cols = st.columns(2, gap="small")
            with column_cols[0]:
                column_name = st.text_input(
                    "Column",
                    value="",
                    placeholder="Optional",
                    help="Optional table column to extract before conversion.",
                    disabled=source_format not in {"csv", "tsv"},
                )
            with column_cols[1]:
                array_key = st.text_input(
                    "Array key",
                    value="",
                    placeholder="Auto",
                    help="Optional NPZ array key.",
                    disabled=source_format != "npz",
                )

            precision_cols = st.columns(2, gap="small")
            with precision_cols[0]:
                use_precision = st.checkbox("Round values", value=False)
            with precision_cols[1]:
                decimal_places = st.number_input(
                    "Decimals",
                    min_value=0,
                    max_value=12,
                    value=4,
                    step=1,
                    disabled=not use_precision,
                )

            conversion_spec = {
                "source_format": source_format,
                "target_format": TARGET_FORMATS[target_label],
                "compress_output": compress_output,
                "data_form": DATA_FORM_LABELS[data_form_label],
                "column_name": column_name.strip() or None,
                "array_key": array_key.strip() or None,
                "decimal_places": int(decimal_places) if use_precision else None,
                "value_name": value_name.strip() or "value",
            }

        with convert_button_slot:
            convert_clicked = st.button(
                "Convert",
                type="primary",
                width="stretch",
            )

        if convert_clicked:
            if not conversion_ready or conversion_spec is None or uploaded_file is None:
                st.error("Upload a supported file before converting.")
            else:
                try:
                    conversion = convert_with_rdata(uploaded_file, **conversion_spec)
                    conversion["spec"] = conversion_spec
                    conversion["source_id"] = upload_token(uploaded_file)
                    st.session_state[LAST_CONVERSION_KEY] = conversion
                except Exception as exc:
                    st.error(f"rdata could not complete this conversion: {exc}")

        conversion = st.session_state.get(LAST_CONVERSION_KEY)
        if conversion is not None and (
            uploaded_file is None
            or conversion.get("source_id") != upload_token(uploaded_file)
        ):
            conversion = None

        if conversion is not None and "spec" in conversion:
            st.markdown(
                """
                <p class="small-muted">
                Converted output is ready.
                </p>
                """,
                unsafe_allow_html=True,
            )
            target_display = str(conversion["target_format"])
            if conversion.get("compressed"):
                target_display = f"{target_display}.gz"
            st.caption(
                f"{conversion['source_format']} -> {target_display} | "
                f"{conversion['data_form']} | values shape {conversion['values_shape']}"
            )

            st.markdown(
                """
                <p class="small-muted">
                Download the converted file.
                </p>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Export")
            export_cols = st.columns([0.68, 0.32], gap="small")
            with export_cols[0]:
                output_file_name = st.text_input(
                    "File name",
                    key="data_management_output_file_name",
                    placeholder="Name (auto)",
                    label_visibility="collapsed",
                )

            payload = conversion["payload"]
            extension = str(conversion.get("extension") or conversion["target_format"])
            mime = conversion["mime"]
            file_stem = safe_stem(output_file_name) or str(conversion["file_stem"])
            file_name = f"{file_stem}.{extension}"
            key_extension = extension.replace(".", "_")
            compressed_suffix = "gz" if conversion.get("compressed") else "plain"
            download_key = (
                f"data_management_download_{file_stem}_{key_extension}_"
                f"{compressed_suffix}"
            )
            with export_cols[1]:
                st.download_button(
                    "Download",
                    data=payload,
                    file_name=file_name,
                    mime=mime,
                    key=download_key,
                    on_click="ignore",
                    width="stretch",
                )
