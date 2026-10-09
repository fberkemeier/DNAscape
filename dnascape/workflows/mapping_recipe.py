"""Reproducible Mapping recipes using the same bounded scientific workflows.

Recipes are ordinary dictionaries: input paths, import choices, region controls,
route settings and batch choices can all be edited without the web application.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .batch import (
    BatchSourceSummary,
    CombinationSource,
    combination_names,
    plan_batch_sources,
    plan_combination_indices,
    unique_dataset_names,
)
from .mapping import (
    CancelChecker,
    MappingRequest,
    MappingResult,
    ProgressReporter,
    materialize_track_upload,
    run_mapping,
    validate_mapping_request,
)
from .mapping_catalog import DataForm, mapping_route, mapping_target
from .result_naming import safe_filename_stem, unique_compact_export_stems
from .uploads import UploadInspection, inspect_upload, materialize_upload

_SCALAR_UNITS = {"firing_rate": "min\u207b\u00b9 kb\u207b\u00b9", "fork_speed": "kb/min"}
_MAX_MAPPING_CASES = 50_000

@dataclass(frozen=True, slots=True)
class _PreparedBatchInput:
    """One route input materialized as one or several independent datasets."""

    key: str
    label: str
    form: DataForm
    values: tuple[Any, ...]
    prewindowed: tuple[bool, ...]
    filenames: tuple[str, ...] = ()
    detected_forms: tuple[str | None, ...] = ()
    components: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _MappingBatchCase:
    """One fully resolved request in a pairwise or Cartesian Mapping run."""

    key: str
    name: str
    export_stem: str
    item_indices: tuple[int, ...]
    short_name: str = ""
    short_export_stem: str = ""


@dataclass(frozen=True, slots=True)
class _NamedMappingResult:
    """A completed mapping result paired with its stable dataset label."""

    key: str
    name: str
    export_stem: str
    result: MappingResult
    short_name: str = ""
    short_export_stem: str = ""


def _friendly_input_key(key: str) -> str:
    return key.removesuffix("_xt")


def _inspection_profile_form(inspection: UploadInspection) -> str | None:
    """Return one file's independently detected profile form."""

    if inspection.source_format in {"bedgraph", "bigwig"}:
        return "x"
    detected = str(inspection.inferred_form)
    return detected if detected in {"x", "t", "xt"} else None


def _format_mapping_component(value: float, *, units: str = "") -> str:
    """Format a scalar compactly for target-first result labels."""

    numeric = f"{float(value):.10g}"
    return f"{numeric} {units}" if units else numeric


def _batch_detected_form(inspection: UploadInspection, selected_form: DataForm) -> str | None:
    """Resolve the X/T vector ambiguity while retaining true XT mismatches."""

    detected = _inspection_profile_form(inspection)
    if selected_form in {"x", "t"} and detected in {"x", "t"}:
        return str(selected_form)
    return detected


def _prepared_components(item: _PreparedBatchInput) -> tuple[str, ...]:
    """Return one deterministic naming component for every prepared value."""

    if len(item.components) == len(item.values):
        return item.components
    if len(item.filenames) == len(item.values):
        return unique_dataset_names(item.filenames)
    if item.form == "scalar":
        units = _SCALAR_UNITS.get(_friendly_input_key(item.key), "")
        return tuple(
            _format_mapping_component(float(value), units=units) for value in item.values
        )
    return tuple(f"Dataset {index}" for index in range(1, len(item.values) + 1))


def _validate_mapping_source_metadata(inputs: Sequence[_PreparedBatchInput]) -> None:
    """Validate every source independently before planning cross-source cases."""

    for item in inputs:
        if not item.values:
            raise ValueError(f"{item.label} must contain at least one value or file.")
        if len(item.prewindowed) != len(item.values):
            raise ValueError(f"{item.label} preprocessing metadata are incomplete.")
        if item.filenames and len(item.filenames) != len(item.values):
            raise ValueError(f"{item.label} upload metadata are incomplete.")
        if item.filenames and len(item.detected_forms) != len(item.filenames):
            raise ValueError(f"{item.label} inspection metadata are incomplete.")
        if item.form == "xt":
            incompatible = tuple(
                filename
                for filename, detected in zip(
                    item.filenames,
                    item.detected_forms,
                    strict=True,
                )
                if detected not in {None, "xt"}
            )
            if incompatible:
                names = ", ".join(incompatible)
                raise ValueError(
                    f"{item.label} expects one complete space-time matrix per dataset; "
                    f"these files are not space-time matrices: {names}."
                )
        if item.filenames:
            plan_batch_sources(
                (
                    BatchSourceSummary(
                        label=item.label,
                        filenames=item.filenames,
                        forms=item.detected_forms,
                        shapes=tuple(
                            tuple(int(size) for size in np.asarray(value).shape)
                            for value in item.values
                        ),
                    ),
                ),
                require_matching_shapes=False,
            )


def _plan_mapping_cases(
    inputs: Sequence[_PreparedBatchInput],
    *,
    target_label: str,
    combination_mode: str = "pairwise",
) -> tuple[_MappingBatchCase, ...]:
    """Resolve pairwise or full Cartesian Mapping cases with target-first names."""

    _validate_mapping_source_metadata(inputs)
    mode = "all" if combination_mode == "all" else "pairwise"
    source_sizes = tuple(len(item.values) for item in inputs)
    if mode == "all":
        projected = math.prod(source_sizes) if source_sizes else 1
    else:
        projected = max(source_sizes, default=1)
    if projected > _MAX_MAPPING_CASES:
        raise ValueError(
            f"These inputs would create {projected:,} mapping runs. "
            f"Reduce the inputs or use at most {_MAX_MAPPING_CASES:,} combinations."
        )
    combinations = plan_combination_indices(source_sizes, mode=mode)
    sources = tuple(
        CombinationSource(
            item.label,
            _prepared_components(item),
            include_singleton_in_short=item.form != "scalar",
        )
        for item in inputs
    )
    identities = combination_names(
        target_label,
        sources,
        combinations,
        include_singletons=True,
    )
    full_stems = unique_compact_export_stems(
        tuple(identity.full for identity in identities),
        fallback="mapping-result",
        maximum=150,
    )
    short_stems = unique_compact_export_stems(
        tuple(identity.short for identity in identities),
        fallback="mapping-result",
        maximum=150,
    )
    cases: list[_MappingBatchCase] = []
    for index, (identity, export_stem, short_export_stem, indices) in enumerate(
        zip(
            identities,
            full_stems,
            short_stems,
            combinations,
            strict=True,
        ),
        start=1,
    ):
        cases.append(
            _MappingBatchCase(
                key=f"mapping-case-{index}",
                name=identity.full,
                export_stem=export_stem,
                item_indices=tuple(indices),
                short_name=identity.short,
                short_export_stem=short_export_stem,
            )
        )
    return tuple(cases)


def _validate_mapping_case_axes(
    inputs: Sequence[_PreparedBatchInput],
    item_indices: Sequence[int],
    dataset_name: str,
) -> None:
    """Validate time axes; spatial inputs are aligned by the Mapping service."""

    if len(inputs) != len(item_indices):
        raise ValueError("A Mapping case must choose one item from every input.")
    temporal: dict[str, int] = {}
    for prepared, item_index in zip(inputs, item_indices, strict=True):
        value = prepared.values[item_index]
        array = np.asarray(value)
        if prepared.form == "t":
            temporal[prepared.label] = int(array.size)
        elif prepared.form == "xt" and array.ndim == 2:
            temporal[prepared.label] = int(array.shape[1])
    # Uploaded chromosome profiles and shorter local outputs can have different
    # spatial lengths. The service applies chromosome Start/End and requires
    # every input to cover the selected width before validating the grid.
    if len(set(temporal.values())) > 1:
        detail = ", ".join(f"{label}: {size:,}" for label, size in temporal.items())
        raise ValueError(f"Dataset {dataset_name!r} has inconsistent time points ({detail}).")


# Public names retain the field layout used by the web application's results.
PreparedMappingInput = _PreparedBatchInput
MappingBatchCase = _MappingBatchCase
NamedMappingResult = _NamedMappingResult
plan_mapping_cases = _plan_mapping_cases
validate_mapping_case_axes = _validate_mapping_case_axes


def _number(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    return numeric


def _optional_number(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _load_prepared_input(source: Mapping[str, Any], base_path: Path) -> dict[str, np.ndarray]:
    path = base_path / str(source["prepared_path"])
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def _inspect_recipe_files(
    source: Mapping[str, Any],
    base_path: Path,
    form: DataForm,
    cancel: CancelChecker,
) -> tuple[tuple[Mapping[str, Any], UploadInspection], ...]:
    files: list[tuple[Mapping[str, Any], UploadInspection]] = []
    for file in source.get("files", ()):
        cancel()
        path = base_path / str(file["path"])
        format_hint = str(file.get("source_format") or "auto").casefold()
        if format_hint not in {"bedgraph", "bigwig"}:
            format_hint = "auto"
        inspection = inspect_upload(
            str(file.get("name") or path.name),
            path.read_bytes(),
            expected_form=form,
            source_format=format_hint,
        )
        files.append((file, inspection))
    if not files:
        raise ValueError("Add at least one input file to this mapping recipe.")
    return tuple(files)


def _track_options(
    source: Mapping[str, Any],
    file: Mapping[str, Any],
    inspection: UploadInspection,
    *,
    start_kb: float | None,
    end_kb: float | None,
    resolution_x: float,
    whole_genome: bool,
) -> tuple[dict[str, Any], bool]:
    controls = dict(source.get("track") or {})
    chromosome = str(controls.get("chromosome") or "").strip() or None
    if inspection.source_format == "bigwig" and chromosome is None:
        raise ValueError("Choose a chromosome before mapping BigWig data.")
    use_region_window = not whole_genome and start_kb is not None and end_kb is not None
    if whole_genome:
        start_bp = end_bp = None
        bin_size = int(round(resolution_x * 1_000))
    elif use_region_window:
        assert start_kb is not None and end_kb is not None
        start_bp = int(round(start_kb * 1_000))
        end_bp = int(round(end_kb * 1_000))
        bin_size = int(round(resolution_x * 1_000))
    else:
        start_bp = controls.get("start")
        end_bp = controls.get("end")
        bin_size = controls.get("bin_size", 1_000)
    if int(bin_size) <= 0:
        raise ValueError("Spatial resolution must span at least one base pair.")
    column = controls.get("value_column")
    if column is None or str(column).strip() == "":
        selected = file.get("column")
        column = None if selected is None or selected == "__all__" else selected
    return {
        "chromosome": chromosome,
        "start": start_bp,
        "end": end_bp,
        "bin_size": bin_size,
        "aggregation": str(controls.get("aggregation") or "mean"),
        "value_column": column,
        "missing": str(controls.get("missing") or "nan"),
        "fill_value": controls.get("fill_value", 0.0),
        "interpolation": str(controls.get("interpolation") or "none"),
        "smooth_window": controls.get("smooth_window", 1),
        "normalize": str(controls.get("normalize") or "none"),
    }, use_region_window


def _prepare_recipe_input(
    key: str,
    source: Mapping[str, Any],
    label: str,
    *,
    base_path: Path,
    prepared: Mapping[str, np.ndarray] | None,
    missing_policy: str,
    start_kb: float | None,
    end_kb: float | None,
    resolution_x: float,
    whole_genome: bool,
    cancel: CancelChecker,
) -> PreparedMappingInput:
    form = cast(DataForm, str(source["form"]))
    mode = str(source.get("mode") or ("constant" if form == "scalar" else "upload"))
    if mode in {"constant", "constants"}:
        raw = source.get("values", source.get("value"))
        supplied = raw if mode == "constants" else (raw,)
        values = tuple(_number(value, label) for value in supplied)
        if not values:
            raise ValueError(f"Add at least one value for {label.lower()}.")
        if mode == "constants" and any(value < 0 for value in values):
            raise ValueError(f"{label} values cannot be negative.")
        units = _SCALAR_UNITS.get(_friendly_input_key(key), "")
        return PreparedMappingInput(
            key, label, form, values, (False,) * len(values),
            components=tuple(_format_mapping_component(value, units=units) for value in values),
        )
    if mode == "regions":
        raw = source.get("values", ())
        if isinstance(raw, str):
            try:
                raw = [
                    [float(value) for value in line.replace("\t", ",").split(",")[-2:]]
                    for line in raw.splitlines() if line.strip()
                ]
            except ValueError as exc:
                raise ValueError(f"{label}: enter one start/end pair per line.") from exc
        values_array = np.asarray(raw, dtype=float)
        if values_array.size == 0:
            raise ValueError(f"Enter at least one region for {label.lower()}.")
        return PreparedMappingInput(key, label, form, (values_array,), (False,), components=("Manual",))
    if mode == "prepared":
        if prepared is None:
            raise ValueError(f"{label} requires a prepared DNAscape result.")
        return PreparedMappingInput(key, label, form, (prepared,), (False,), components=("Manual",))
    if mode != "upload":
        raise ValueError(f"Unknown mapping input mode {mode!r}.")

    files = _inspect_recipe_files(source, base_path, form, cancel)
    expand_columns = form in {"x", "t"} and any(
        file.get("column") == "__all__" for file, _inspection in files
    )
    use_track = source.get("track") is not None and all(
        inspection.source_format in {"bedgraph", "bigwig"} for _file, inspection in files
    )
    if use_track:
        track_controls = dict(source.get("track") or {})
        explicit_column = str(track_controls.get("value_column") or "").strip()
        selected_columns = {
            str(file["column"]) for file, _inspection in files
            if file.get("column") is not None and file["column"] != "__all__"
        }
        if not explicit_column and len(selected_columns) > 1:
            raise ValueError("Choose the same bedGraph value column for every uploaded time point.")
        if not explicit_column and selected_columns:
            source = {**source, "track": {**track_controls, "value_column": next(iter(selected_columns))}}
    values_list: list[Any] = []
    prewindowed: list[bool] = []
    filenames: list[str] = []
    detected_forms: list[str | None] = []
    components: list[str] = []
    for file, inspection in files:
        cancel()
        columns: tuple[str | None, ...] = (
            tuple(inspection.numeric_columns)
            if expand_columns and file.get("column") == "__all__"
            else (cast(str | None, file.get("column")),)
        )
        for column in columns:
            selected_column = None if column == "__all__" else column
            if use_track and not expand_columns:
                track_options, is_prewindowed = _track_options(
                    source, file, inspection, start_kb=start_kb, end_kb=end_kb,
                    resolution_x=resolution_x, whole_genome=whole_genome,
                )
                values = materialize_track_upload(inspection, **track_options)
            else:
                values = materialize_upload(
                    inspection, column=selected_column, array_key=file.get("array_key"),
                    missing_policy=missing_policy,
                )
                is_prewindowed = False
            values_list.append(values)
            prewindowed.append(is_prewindowed)
            detected_forms.append(_batch_detected_form(inspection, form))
            if expand_columns:
                dataset_label = unique_dataset_names((inspection.name,))[0]
                column_label = str(column or inspection.default_column or "values")
                expanded = file.get("column") == "__all__"
                component = f"{dataset_label} · {column_label}" if expanded else dataset_label
                synthetic_stem = safe_filename_stem(
                    f"{dataset_label}-{column_label}" if expanded else dataset_label,
                    fallback=f"dataset-{len(values_list)}",
                )
                filenames.append(f"{synthetic_stem}.profile")
                components.append(component)
            else:
                filenames.append(inspection.name)
    if not expand_columns:
        components.extend(unique_dataset_names(filenames))
    return PreparedMappingInput(
        key, label, form, tuple(values_list), tuple(prewindowed), tuple(filenames),
        tuple(detected_forms), tuple(components),
    )


def prepare_mapping_recipe(
    recipe: Mapping[str, Any],
    *,
    base_path: str | Path = Path("."),
    cancel: CancelChecker | None = None,
) -> tuple[tuple[MappingBatchCase, MappingRequest], ...]:
    """Load editable recipe inputs and build the application's exact requests.

    Uploaded files are inspected and imported afresh using their current column,
    missing-value and track choices. No browser state or application imports are
    required. Every relative input path is resolved against ``base_path``.
    """
    check_cancelled = cancel or (lambda: None)
    check_cancelled()
    root = Path(base_path)
    route = mapping_route(str(recipe["method"]))
    settings = dict(recipe.get("settings", {}))
    for key, override in recipe.get("setting_overrides", {}).items():
        if bool(override.get("enabled", False)):
            settings[key] = override.get("value")
        else:
            settings.pop(key, None)
    region = recipe.get("region", {})
    start_kb = _optional_number(region.get("start_kb"), "Region start")
    end_kb = _optional_number(region.get("end_kb"), "Region end")
    resolution_value = settings.get("resolution")
    if resolution_value is None:
        resolution_value = route.setting_defaults_dict.get("resolution", 1.0)
    resolution_x = _number(1.0 if resolution_value is None else resolution_value, "Spatial resolution")
    sources = recipe.get("inputs", {})
    prepared_values: dict[str, dict[str, np.ndarray]] = {}
    for key, source in sources.items():
        if not source.get("enabled", True) or source.get("mode") != "prepared":
            continue
        check_cancelled()
        prepared_values[key] = _load_prepared_input(source, root)
        axis = np.asarray(prepared_values[key].get("x_array", ()), dtype=float)
        if axis.size > 1:
            resolution_x = float(axis[1] - axis[0])
    whole_genome = bool(region.get("whole_genome", False))
    labels = {item.key: item.label for item in route.inputs}
    labels.update({_friendly_input_key(item.key): item.label for item in route.inputs})
    prepared_inputs = tuple(
        _prepare_recipe_input(
            key, source, labels.get(key, key.replace("_", " ").title()),
            base_path=root, prepared=prepared_values.get(key),
            missing_policy=str(region.get("missing_policy") or "error"),
            start_kb=start_kb, end_kb=end_kb, resolution_x=resolution_x,
            whole_genome=whole_genome, cancel=check_cancelled,
        )
        for key, source in sources.items() if source.get("enabled", True)
    )
    has_multiple_constants = any(source.get("mode") == "constants" for source in sources.values())
    cases = plan_mapping_cases(
        prepared_inputs, target_label=mapping_target(route.target_key).label,
        combination_mode=str(recipe.get("combination_mode") or "pairwise")
        if has_multiple_constants else "pairwise",
    )
    requests: list[tuple[MappingBatchCase, MappingRequest]] = []
    for case in cases:
        check_cancelled()
        validate_mapping_case_axes(prepared_inputs, case.item_indices, case.name)
        selected = tuple(zip(prepared_inputs, case.item_indices, strict=True))
        requests.append((case, MappingRequest(
            route=route,
            inputs={item.key: item.values[index] for item, index in selected},
            input_forms={item.key: item.form for item in prepared_inputs},
            settings=settings,
            target_shape=recipe.get("target_shape"),
            region_start_kb=start_kb,
            region_end_kb=end_kb,
            resolution_x=resolution_x,
            resolution_t=_optional_number(settings.get("resolution_time"), "Temporal resolution"),
            whole_genome=whole_genome,
            prewindowed_inputs=frozenset(item.key for item, index in selected if item.prewindowed[index]),
        )))
    return tuple(requests)


def run_mapping_recipe(
    recipe: Mapping[str, Any],
    *,
    base_path: str | Path = Path("."),
    report: ProgressReporter | None = None,
    cancel: CancelChecker | None = None,
) -> tuple[NamedMappingResult, ...]:
    """Run a saved Mapping page configuration without loading the web app.

    All requests are validated before the first mapping run, matching the page's
    batch behaviour. Scientific output arrays and their axes use the same
    ``MappingResult`` objects returned to the app.
    """
    check_cancelled = cancel or (lambda: None)
    requests = prepare_mapping_recipe(recipe, base_path=base_path, cancel=check_cancelled)
    for _case, request in requests:
        check_cancelled()
        validate_mapping_request(request, check_cancelled)
    completed: list[NamedMappingResult] = []
    total = len(requests)
    for index, (case, request) in enumerate(requests):
        check_cancelled()

        def report_case(value: float, message: str, case_index: int = index) -> None:
            if report is not None:
                report((case_index + value) / total, message)

        result = run_mapping(request, report_case, check_cancelled)
        completed.append(NamedMappingResult(
            case.key, case.name, case.export_stem, result, case.short_name, case.short_export_stem,
        ))
    return tuple(completed)


__all__ = [
    "MappingBatchCase", "NamedMappingResult", "PreparedMappingInput",
    "plan_mapping_cases", "prepare_mapping_recipe", "run_mapping_recipe",
    "validate_mapping_case_axes",
]
