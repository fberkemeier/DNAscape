"""Focused behavior tests for Plotting-page profile grouping."""

from __future__ import annotations

import numpy as np
import pytest

from app.pages.plotting import (
    _combined_series_compatible,
    _plot_groups,
    _PlotSeries,
    _series_compatibility_error,
)


def _series(name: str, form: str) -> _PlotSeries:
    values = (
        np.ones((3, 2), dtype=float)
        if form == "xt"
        else np.arange(3, dtype=float)
    )
    return _PlotSeries(name=name, form=form, values=values)


def test_combined_layout_only_accepts_like_for_like_1d_profiles() -> None:
    spatial = (_series("a", "x"), _series("b", "x"))
    temporal = (_series("a", "t"), _series("b", "t"))
    mixed = (_series("space", "x"), _series("time", "t"))

    assert _combined_series_compatible(spatial)
    assert _combined_series_compatible(temporal)
    assert not _combined_series_compatible(mixed)
    assert [[item.name for item in group] for group in _plot_groups(spatial, "Combined")] == [
        ["a", "b"]
    ]
    with pytest.raises(ValueError, match="need separate plots"):
        _plot_groups(mixed, "Combined")


def test_separate_layout_accepts_1d_mixtures_and_space_with_spacetime() -> None:
    space_time = (_series("space", "x"), _series("time", "t"))
    space_heatmap = (_series("space", "x"), _series("map", "xt"))

    assert [group[0].name for group in _plot_groups(space_time, "Separate")] == [
        "space",
        "time",
    ]
    assert [group[0].name for group in _plot_groups(space_heatmap, "Separate")] == [
        "space",
        "map",
    ]


def test_time_and_spacetime_are_rejected_with_concise_guidance() -> None:
    incompatible = (_series("time", "t"), _series("map", "xt"))

    assert _series_compatibility_error(incompatible) == (
        "Time profiles cannot be mixed with space-time heatmaps. "
        "Remove one of these data types before plotting."
    )
    with pytest.raises(ValueError, match="cannot be mixed"):
        _plot_groups(incompatible, "Separate")
