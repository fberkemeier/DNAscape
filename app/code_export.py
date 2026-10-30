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
    if controls.upload is not None and settings.selection == "profile":
        settings = replace(settings, profile=None, upload=controls.upload)
    if settings.enabled:
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


def build_python(snapshot: CodeSnapshot) -> ExportArtifact:
    """A single entry point, editable folders and a plain configuration dictionary."""
    workflow = snapshot.workflow
    if workflow not in {"data_manager", "simulation", "plotting"}:
        raise ValueError("Choose a supported code workflow.")
    inputs = "\n".join(f"#   {path!r}" for path, _data in snapshot.files) or "#   No external files: constants or prepared arrays are included below."
    source = f'''# DNAscape {workflow.replace("_", " ").title()}
# Run with the DNAscape package used to export this script installed:
#   python {workflow}.py
# Edit INPUT_DIRECTORY and the filenames below to use your local data.
# Missing uploads use placeholder filenames; replace them or supply those files:
{inputs}

from pathlib import Path
from math import inf, nan

from dnascape.workflows.{"analysis_recipe" if workflow == "plotting" else "simulation_recipe" if workflow == "simulation" else "code_recipes"} import run_{workflow}_recipe

HERE = Path(__file__).resolve().parent
INPUT_DIRECTORY = HERE / "inputs"
OUTPUT_DIRECTORY = HERE / "outputs"

# Edit source paths, selections and options here. Positions and time retain
# the units shown by the app. A seed of None makes stochastic runs independent.
RECIPE = {pprint.pformat(snapshot.recipe, width=95, sort_dicts=False)}


def main():
    results, paths = run_{workflow}_recipe(
        RECIPE, input_directory=INPUT_DIRECTORY, output_directory=OUTPUT_DIRECTORY,
    )
    for path in paths:
        print(f"Saved {{path}}")
    return results


if __name__ == "__main__":
    main()
'''
    return ExportArtifact(source.encode("utf-8"), f"{workflow}.py", "text/x-python; charset=utf-8")
