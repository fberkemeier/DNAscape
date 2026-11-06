"""Small, editable .py downloads; scientific execution stays in DNAscape."""
from __future__ import annotations

import pprint
from dataclasses import dataclass, replace
from pathlib import PurePath

from dnascape.workflows.code_recipes import encode_settings
from dnascape.workflows.exports import ExportArtifact


@dataclass(frozen=True)
class CodeSnapshot:
    workflow: str
    recipe: dict
    files: tuple[tuple[str, bytes], ...] = ()


def placeholder_upload(name="source", *, form="x"):
    """Describe an editable file input without inspecting or inventing data."""
    from dnascape.workflows.uploads import UploadInspection

    extension = "npz" if form == "xt" else "tsv"
    return UploadInspection(
        name=f"{name}.{extension}", content_type="application/octet-stream",
        content=b"", source_format=extension, compressed=False,
        expected_form=form, inferred_form=form,
    )


class CodeInputs:
    """Retain original filenames, deduplicating identical shared source files."""
    def __init__(self):
        self.files = {}

    def upload(self, inspection):
        name = PurePath(inspection.name.replace("\\", "/")).name
        path = name
        index = 2
        while any(existing.casefold() == path.casefold() for existing in self.files):
            existing = next(existing for existing in self.files if existing.casefold() == path.casefold())
            if self.files[existing] == inspection.content:
                return existing
            path = f"source_{index}/{name}"
            index += 1
        self.files[path] = bytes(inspection.content)
        return path

    def encode(self, settings):
        return encode_settings(settings, upload=self.upload)


def filter_code_settings(controls):
    """Reference the original mask upload instead of copying its full position list."""
    settings = controls.snapshot()
    if not settings.enabled:
        # An inactive transform must not require stale mask/modulation uploads.
        return type(settings)()
    if controls.upload is not None and settings.selection == "profile":
        settings = replace(settings, profile=None, upload=controls.upload)
    prefix = controls.marker_prefix.replace("-", "_")
    if settings.selection == "profile" and settings.profile is None and settings.upload is None:
        settings = replace(settings, upload=placeholder_upload(prefix + "_mask_positions", form="regions"),
                           profile_layout="positions")
    if (settings.action == "modulate" and settings.modulation.upload is None
            and settings.modulation.values is None and not settings.modulation.profiles):
        settings = replace(settings, modulation=replace(settings.modulation,
            upload=placeholder_upload(prefix + "_modulation_profile", form="auto")))
    return settings


def statistics_code_settings(controls, *, filename="histogram_statistics"):
    if controls.view.value != "Histogram" or not controls.export_histogram_statistics.value:
        return None
    from app.preview_plotting import histogram_x_limits_from_controls
    exporter = controls._statistics_export
    snapshot = (exporter.snapshot() if exporter is not None else
                dict(statistics=[], filename=filename, file_format="TSV", compression="none"))
    labels = [row["dataset"] for row in snapshot.pop("statistics", [])]
    return dict(snapshot, labels=labels, density=bool(controls.density.value),
                log_x=bool(controls.log_x.value),
                limits=histogram_x_limits_from_controls(controls, log_x=bool(controls.log_x.value)))


def recipe_literal(recipe, groups):
    """Keep one editable value per field, grouped like the workflow page."""
    from textwrap import indent

    lines = ["{"]
    remaining = dict(recipe)
    for label, keys in groups:
        selected = {key: remaining.pop(key) for key in keys if key in remaining}
        if not selected:
            continue
        lines.append(f"    # {label}")
        for key, value in selected.items():
            field = pprint.pformat({key: value}, width=91, sort_dicts=False)[1:-1]
            lines.append(indent(field, "    ") + ",")
    for key, value in remaining.items():
        field = pprint.pformat({key: value}, width=91, sort_dicts=False)[1:-1]
        lines.append(indent(field, "    ") + ",")
    return "\n".join([*lines, "}"])


def render_python(*, workflow, imports, configuration, execution, files, recipe_name="RECIPE"):
    """Use the same portable script shell for every developer download."""
    from textwrap import indent
    from dnascape.workflows.code_recipes import encode_input_files

    embedded = encode_input_files(files)
    encoded_lines = ["{"]
    for path, data in embedded.items():
        encoded_lines.append(f"    {path!r}: (")
        encoded_lines.extend(f"        {data[index:index + 80]!r}" for index in range(0, len(data), 80))
        encoded_lines.append("    ),")
    embedded_literal = "\n".join([*encoded_lines, "}"]) if embedded else "{}"
    manifest = "\n".join(
        f"#   {path!r}" + (" (included)" if data else " (supply this file)")
        for path, data in files
    ) or "#   No external files: constants or prepared arrays are included below."
    title = "Analysis" if workflow == "plotting" else workflow.replace("_", " ").title()
    seed_hint = "\n# An integer seed reproduces simulations; None gives independent runs." if workflow == "simulation" else ""
    source = f'''# DNAscape {title}
# Run with the DNAscape package used to export this script installed:
#   python {workflow}.py
# Original uploads are included; existing local files take precedence.
# Missing uploads use placeholder filenames; replace them or supply those files:
{manifest}

from math import inf, nan
from pathlib import Path

from dnascape.workflows.code_recipes import prepare_input_files
{imports}

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
INPUT_DIRECTORY = HERE / "inputs"
OUTPUT_DIRECTORY = HERE / "outputs"
USE_EMBEDDED_INPUTS = True  # Restore missing originals; never overwrite local files.

# Edit paths, columns/arrays, constants and options below. Units match the app.{seed_hint}
{configuration}


def main(recipe=None, *, input_directory=None, output_directory=None):
    """Run here, or call main(recipe=...) for a parameter sweep or notebook."""
    recipe = {recipe_name} if recipe is None else recipe
    input_directory = INPUT_DIRECTORY if input_directory is None else Path(input_directory)
    output_directory = OUTPUT_DIRECTORY if output_directory is None else Path(output_directory)
    if USE_EMBEDDED_INPUTS:
        prepare_input_files(EMBEDDED_INPUTS, input_directory, recipe=recipe)
{indent(execution, "    ")}
    for path in paths:
        print(f"Saved {{path}}")
    return results


# Original source bytes, compressed for portability. Edit the recipe above;
# set USE_EMBEDDED_INPUTS = False to require local files for uploaded sources.
EMBEDDED_INPUTS = {embedded_literal}


if __name__ == "__main__":
    main()
'''
    return ExportArtifact(source.encode("utf-8"), f"{workflow}.py", "text/x-python; charset=utf-8")


def build_python(snapshot: CodeSnapshot) -> ExportArtifact:
    """A complete script with editable inputs, workflow options and outputs."""
    workflow = snapshot.workflow
    modules = {"data_manager": "code_recipes", "simulation": "simulation_recipe", "plotting": "analysis_recipe"}
    if workflow not in modules:
        raise ValueError("Choose a supported code workflow.")
    groups = (
        ("Inputs: paths, dependence, columns/arrays and constants.", ("sources", "inputs", "profiles")),
        ("Settings: preparation, region, transforms and calculation.", (
            "settings", "region_end_kb", "automatic_region_end", "combination_mode",
            "combined_formula", "combined_name", "data_transforms", "transformations", "methods",
        )),
        ("Outputs: filenames, formats, compression, statistics and figures.", (
            "exports", "export", "statistics", "data_exports", "mask_exports", "report", "report_names",
            "plots", "plot_dpi", "preview_points",
        )),
    )
    return render_python(
        workflow=workflow, files=snapshot.files,
        imports=f"from dnascape.workflows.{modules[workflow]} import run_{workflow}_recipe",
        configuration="RECIPE = " + recipe_literal(snapshot.recipe, groups),
        execution=f"results, paths = run_{workflow}_recipe(\n"
                  "    recipe, input_directory=input_directory, output_directory=output_directory,\n)",
    )
