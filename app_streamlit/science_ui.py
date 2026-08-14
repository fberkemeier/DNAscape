"""Reusable Streamlit controls for mapping and simulation inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import streamlit as st

from app_streamlit.preview import memory_label
from app_streamlit.state import persistent_widget_key
from app_streamlit.workflow_ui import DatasetView, render_profile_view, view_axis, view_series


@dataclass(frozen=True)
class FieldInput:
    """A constant or non-owning canonical view selected by the user."""

    feature: str
    source_id: str
    name: str
    values: float | np.ndarray
    data_form: str
    axes: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self.values).shape)


def _stable_choice(key: str, options: Sequence[str], default: str) -> None:
    if not options:
        raise ValueError("This selector has no available options.")
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]


def _select_view_values(
    namespace: str,
    view: DatasetView,
) -> tuple[np.ndarray, str, str]:
    values = view.values
    data_form = view.data_form
    selected_label = ""
    if values.size == 0 or any(axis == 0 for axis in values.shape):
        raise ValueError("The selected profile is empty.")
    if values.ndim == 2 and data_form in {"space", "table"}:
        labels = view.labels if len(view.labels) == values.shape[1] else tuple(
            f"value_{index + 1}" for index in range(values.shape[1])
        )
        key = persistent_widget_key(namespace, "profile_column")
        _stable_choice(key, labels, labels[0])
        selected_label = st.selectbox("Profile column", labels, key=key)
        values = view_series(view, selected_label)
        if data_form == "table":
            data_form = "space"
    elif data_form == "time":
        values = values.reshape(-1)
    return np.asarray(values), data_form, selected_label


def render_field_input(
    namespace: str,
    feature: str,
    label: str,
    *,
    allow_constant: bool,
    allow_profile: bool = True,
    constant_default: float,
    constant_min: float,
    constant_step: float,
    units: str,
    default_mode: str = "Profile",
    help_text: str = "",
) -> FieldInput | None:
    """Render a reusable constant/profile selector without parsing on reruns."""

    modes = tuple(
        mode
        for mode, allowed in (("Constant", allow_constant), ("Profile", allow_profile))
        if allowed
    )
    if not modes:
        raise ValueError("At least one field-input mode must be enabled.")
    if not np.isfinite(constant_default) or not np.isfinite(constant_min):
        raise ValueError("Constant defaults and bounds must be finite.")
    if not np.isfinite(constant_step) or constant_step <= 0:
        raise ValueError("Constant input step must be finite and positive.")
    if constant_default < constant_min:
        raise ValueError("Constant default cannot be below its minimum.")
    mode_key = persistent_widget_key(namespace, "input_mode")
    _stable_choice(mode_key, modes, default_mode)
    mode = st.radio(
        f"{label} input",
        modes,
        horizontal=True,
        key=mode_key,
        help=help_text or None,
    )
    if mode == "Constant":
        key = persistent_widget_key(namespace, "constant")
        if key not in st.session_state:
            st.session_state[key] = float(constant_default)
        value = float(
            st.number_input(
                f"{label} ({units})",
                min_value=float(constant_min),
                step=float(constant_step),
                key=key,
            )
        )
        return FieldInput(
            feature=feature,
            source_id=f"constant:{feature}:{value:.17g}",
            name=f"Constant {label}",
            values=value,
            data_form="scalar",
        )

    view = render_profile_view(namespace)
    if view is None:
        st.caption(f"Choose or prepare the {label.lower()} profile to continue.")
        return None
    try:
        values, data_form, selected_label = _select_view_values(namespace, view)
    except ValueError as exc:
        st.error(str(exc))
        return None
    suffix = f":{selected_label}" if selected_label else ""
    st.caption(
        f"{view.name}{' / ' + selected_label if selected_label else ''} - "
        f"{data_form} - {' x '.join(map(str, values.shape)) or 'scalar'} - "
        f"{memory_label(values.nbytes)}"
    )
    return FieldInput(
        feature=feature,
        source_id=f"{view.view_id}{suffix}",
        name=view.name,
        values=values,
        data_form=data_form,
        axes=view.axes,
    )


def spatial_axis_for_input(field: FieldInput, resolution_kb: float) -> np.ndarray | None:
    """Return a normalized kb axis for a spatial field, if it has one."""

    if field.data_form not in {"space", "spacetime", "table"}:
        return None
    values = np.asarray(field.values)
    length = values.shape[0] if values.ndim else 0
    if not length:
        return None
    pseudo_view = DatasetView(
        view_id=field.source_id,
        name=field.name,
        values=values,
        data_form=field.data_form,
        axes=field.axes,
    )
    return view_axis(pseudo_view, length, resolution_kb=resolution_kb)
