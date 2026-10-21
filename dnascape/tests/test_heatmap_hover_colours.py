"""Hovered Z readouts use the raster palette, clipping, inversion, and log scale."""

import base64
import io
import json
import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from dnascape.workflows.preview import PALETTES, build_preview_plot_options


def test_hover_colours_match_rendered_pixels_for_every_palette_and_scale(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute hover callbacks.")
    cases = []
    expected = []
    for palette in PALETTES:
        for log_z in (False, True):
            for invert_z in (False, True):
                values = np.asarray([[.001, .01, 1., 100., 1000.]]) if log_z else np.asarray([[-1., 0., .5, 1., 2.]])
                options = build_preview_plot_options(
                    plot_type="Heatmap", title="Kinetics", x_label="Position", y_label="Time",
                    series_label="Firing <rate>", color="#b03412", palette=palette,
                    histogram_bins=20, density=False, show_legend=False, show_colorbar=True,
                    log_y=False, invert_y=False, log_z=log_z, invert_z=invert_z,
                    y_limits=None,
                    values=values, x_axis=np.arange(5.), y_axis=np.asarray([0.]),
                    z_limits=(.01, 100.) if log_z else (0., 1.),
                )
                encoded = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", options["series"][0][":renderItem"])
                pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded[1]))).convert("RGB"))
                cases.append({"formatter": options["tooltip"][":formatter"], "points": options["series"][1]["data"]})
                expected.append(pixels[0].tolist())
    callbacks = tmp_path / "colours.json"
    callbacks.write_text(json.dumps(cases), encoding="utf-8")
    completed = subprocess.run([node, "-e", """
const fs = require('node:fs');
const cases = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
console.log(JSON.stringify(cases.map(item => {
  const formatter = Function('return (' + item.formatter + ')')();
  return item.points.map(value => formatter({value, marker: 'OLD_FIXED_MARKER'}));
})));
""", str(callbacks)], capture_output=True, text=True, check=True)
    actual = json.loads(completed.stdout)
    for previews, pixels in zip(actual, expected, strict=True):
        for text, rgb in zip(previews, pixels, strict=True):
            assert f"background:rgb({','.join(map(str, rgb))})" in text
            assert all(label not in text for label in ("Firing", "Position", "Time", "<br>"))
            assert re.fullmatch(r"-?[\d.]+(?:e[+-]?\d+)?", re.sub(r"<[^>]+>", "", text))
            assert "OLD_FIXED_MARKER" not in text
            foreground = re.search(r";color:(#[0-9a-f]{6});", text)[1]
            linear = [value / 255 / 12.92 if value / 255 <= .04045 else ((value / 255 + .055) / 1.055)**2.4 for value in rgb]
            luminance = sum(weight * value for weight, value in zip((.2126, .7152, .0722), linear))
            contrast = (luminance + .05) / .05 if foreground == "#000000" else 1.05 / (luminance + .05)
            assert contrast >= 4.5
