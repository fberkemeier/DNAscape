"""Regression coverage for bounded, defensive app plotting."""

from __future__ import annotations

import base64
import importlib
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np

from app_streamlit.plot_service import (
    MAX_HISTOGRAM_BINS,
    heatmap_figure,
    histogram_figure,
    line_figure,
    release_figure,
)
from app_streamlit.preview import estimated_array_bytes, preview_indices
from app_streamlit.preview_ui import render_image_preview
from app_streamlit.workflow_ui import DatasetView

plotting_page = importlib.import_module("app_streamlit.pages.4_Plotting")


def _config(plot_type: str, shape: tuple[int, ...]) -> dict[str, object]:
    rows = shape[0] if shape else 1
    columns = shape[1] if len(shape) == 2 else 1
    return {
        "plot_type": plot_type,
        "series": None,
        "comparison_view_id": "",
        "comparison_color": "#d97757",
        "normalization": "None",
        "row_start": 0,
        "row_end": rows,
        "column_start": 0,
        "column_end": columns,
        "title": "Test",
        "x_label": "X",
        "y_label": "Y",
        "colorbar_label": "Value",
        "spatial_step_kb": 1.0,
        "time_step_min": 1.0,
        "color": "#2c7f77",
        "cmap": "viridis",
        "line_width": 1.8,
        "bins": 60,
        "show_grid": True,
        "show_colorbar": True,
        "log_y": False,
        "invert_y": False,
        "x_limits": None,
        "y_limits": None,
        "vmin": None,
        "vmax": None,
    }


class PreviewHelperTests(unittest.TestCase):
    def test_preview_indices_are_bounded_and_keep_endpoints(self) -> None:
        indices = preview_indices(1_000_000, 5_000)
        self.assertLessEqual(indices.size, 5_000)
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[-1]), 999_999)

    def test_array_byte_estimate_does_not_overflow_int64(self) -> None:
        self.assertEqual(
            estimated_array_bytes((2**32, 2**32), np.uint8),
            2**64,
        )


class PlotServiceTests(unittest.TestCase):
    def test_line_preview_is_bounded(self) -> None:
        values = np.linspace(0.0, 1.0, 50_000)
        figure, axes = line_figure(values, values)
        try:
            self.assertLessEqual(np.asarray(axes.lines[0].get_xdata()).size, 5_000)
        finally:
            release_figure(figure)

    def test_line_rejects_misaligned_or_complex_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            line_figure(np.arange(3), np.arange(2))
        with self.assertRaisesRegex(ValueError, "real numeric"):
            line_figure(np.arange(2), np.array([1 + 2j, 3 + 4j]))

    def test_heatmap_preview_is_bounded(self) -> None:
        values = np.arange(1_200 * 600, dtype=float).reshape(1_200, 600)
        figure, axes = heatmap_figure(values)
        try:
            self.assertEqual(
                tuple(np.asarray(axes.images[0].get_array()).shape),
                (900, 420),
            )
        finally:
            release_figure(figure)

    def test_one_cell_heatmap_gets_nonzero_coordinate_extent(self) -> None:
        figure, axes = heatmap_figure(
            np.array([[1.0]]),
            x_values=np.array([10.0]),
            y_values=np.array([20.0]),
            show_colorbar=False,
        )
        try:
            self.assertEqual(tuple(axes.images[0].get_extent()), (9.5, 10.5, 19.5, 20.5))
        finally:
            release_figure(figure)

    def test_heatmap_rejects_nonmonotonic_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly monotonic"):
            heatmap_figure(
                np.ones((2, 3)),
                x_values=np.array([0.0, 2.0, 1.0]),
                y_values=np.array([0.0, 1.0]),
            )

    def test_histogram_caps_hostile_bin_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "between"):
            histogram_figure(np.arange(10), bins=MAX_HISTOGRAM_BINS + 1)


class PlottingPageTests(unittest.TestCase):
    def test_build_figure_bounds_each_preview_kind(self) -> None:
        line_values = np.linspace(0.0, 1.0, 20_000)
        line_view = DatasetView("line", "line", line_values, "space")
        line_figure_result, line_axes = plotting_page._build_figure(
            line_view,
            _config("Line", line_values.shape),
            full_resolution=False,
        )
        try:
            self.assertLessEqual(line_axes.lines[0].get_xdata().size, 5_000)
        finally:
            release_figure(line_figure_result)

        matrix = np.ones((1_000, 500), dtype=float)
        matrix_view = DatasetView("matrix", "matrix", matrix, "spacetime")
        heatmap_result, heatmap_axes = plotting_page._build_figure(
            matrix_view,
            _config("Heatmap", matrix.shape),
            full_resolution=False,
        )
        try:
            self.assertEqual(
                tuple(heatmap_axes.images[0].get_array().shape),
                (900, 420),
            )
        finally:
            release_figure(heatmap_result)

    def test_build_figure_rejects_unknown_plot_type(self) -> None:
        values = np.arange(10, dtype=float)
        view = DatasetView("line", "line", values, "space")
        with self.assertRaisesRegex(ValueError, "Unsupported plot type"):
            plotting_page._build_figure(
                view,
                _config("Surprise", values.shape),
                full_resolution=False,
            )


class PreviewUiTests(unittest.TestCase):
    def test_empty_download_mapping_avoids_duplicate_data_uri(self) -> None:
        payload = b"png-payload"
        encoded = base64.b64encode(payload).decode("ascii")
        with (
            patch("app_streamlit.preview_ui.st.container", return_value=nullcontext()),
            patch("app_streamlit.preview_ui.st.button"),
            patch("app_streamlit.preview_ui.st.html") as html_output,
        ):
            render_image_preview(
                payload,
                file_name="preview.png",
                shell_key="preview-shell",
                clear_key="preview-clear",
                on_clear=lambda: None,
                downloads={},
            )
        markup = html_output.call_args.args[0]
        self.assertEqual(markup.count(encoded), 1)
        self.assertNotIn("Download preview", markup)

    def test_preview_rejects_empty_payload_before_rendering(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            render_image_preview(
                b"",
                file_name="preview.png",
                shell_key="preview-shell",
                clear_key="preview-clear",
                on_clear=lambda: None,
            )


if __name__ == "__main__":
    unittest.main()
