"""Per-source and combined formula controls for Data Manager."""

from nicegui import ui

class DataTransformControls:
    def __init__(self, sources, *, value_controls=None, marker_prefix="data-manager-transform",
                 source_label="Source", header=None, on_change=None):
        self.sources = sources
        self.marker_prefix = marker_prefix
        self.source_label = source_label
        self.on_change = on_change or (lambda: None)
        self.formulas = {}
        self.combined_formula = None
        self.fields = []
        with ui.expansion("Data transforms", icon="functions", value=False).props(
            "dense"
        ).classes("dn-expansion").mark(marker_prefix):
            with ui.column().classes("dn-expansion-content w-full") as content:
                with ui.row().classes("dn-inline-actions"):
                    self.enabled = ui.switch("Apply transform", value=False).mark(f"{marker_prefix}-enabled")
                    self.combine = ui.switch("Combine", value=False).mark(f"{marker_prefix}-combine")
                if header is not None:
                    header()
                self.hint = ui.label().classes("dn-upload-formats")
                self._render = ui.refreshable(self._render_fields)
                self._render()
                if value_controls is not None:
                    value_controls.move(content)
        self.combine.on_value_change(self.refresh)
        self.enabled.on_value_change(self.on_change)
        self.combine.on_value_change(self.on_change)
        self.refresh()

    @property
    def combining(self):
        return bool(self.enabled.value and self.combine.value and len(self.sources()) > 1)

    def formula_for(self, source_id):
        return str(self.formulas.get(source_id, "source") or "") if self.enabled.value and not self.combining else None

    def refresh(self):
        multiple = len(self.sources()) > 1
        self.combine.set_visibility(multiple)
        if not multiple and self.combine.value:
            self.combine.set_value(False)
        self._render.refresh()

    def _render_fields(self):
        self.fields.clear()
        sources = self.sources()
        combine = bool(self.combine.value and len(sources) > 1)
        self.hint.set_text(
            "Python / NumPy syntax. Use source1, source2, … for the loaded signals; keep their array shape."
            if combine else
            "Python / NumPy syntax. Use source for each signal, e.g. source / np.mean(source). Runs after Configure."
        )
        if self.source_label != "Source":
            self.hint.set_text(self.hint.text.replace("Runs after Configure.", "Keeps the dataset shape."))
        if sources and None in self.formulas:
            self.formulas.setdefault(sources[0][0], self.formulas.pop(None))
        if combine:
            formula = self.combined_formula
            if formula is None:
                formula = " + ".join(f"source{i}" for i in range(1, len(sources) + 1))
            field = ui.textarea("Combined transform formula", value=formula).props(
                "dense autogrow rows=2 spellcheck=false"
            ).classes("dn-field dn-formula-input").mark(f"{self.marker_prefix}-combined-formula")
            field.on_value_change(lambda event: setattr(self, "combined_formula", event.value))
            field.on_value_change(self.on_change)
            self.fields.append(field)
        else:
            for index, (source_id, _name) in enumerate(sources or [(None, "")], 1):
                label = (
                    f"{self.source_label} transform formula" if len(sources) <= 1
                    else f"{self.source_label} {index} transform formula"
                )
                field = ui.textarea(label, value=self.formulas.get(source_id, "source")).props(
                    "dense autogrow rows=2 spellcheck=false"
                ).classes("dn-field dn-formula-input").mark(f"{self.marker_prefix}-formula-{index}")
                field.on_value_change(lambda event, key=source_id: self.formulas.__setitem__(key, event.value))
                field.on_value_change(self.on_change)
                self.fields.append(field)

    def set_locked(self, locked):
        for control in (self.enabled, self.combine, *self.fields):
            control.set_enabled(not locked)
