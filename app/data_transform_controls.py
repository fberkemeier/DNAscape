"""Per-source and combined formula controls for Data Manager."""

from nicegui import ui

class DataTransformControls:
    def __init__(self, sources):
        self.sources = sources
        self.formulas = {}
        self.combined_formula = None
        self.fields = []
        with ui.expansion("Data transforms", icon="functions", value=False).props(
            "dense"
        ).classes("dn-expansion").mark("data-manager-transform"):
            with ui.column().classes("dn-expansion-content w-full"):
                with ui.row().classes("dn-inline-actions"):
                    self.enabled = ui.switch("Apply transform", value=False).mark("data-manager-transform-enabled")
                    self.combine = ui.switch("Combine", value=False).mark("data-manager-transform-combine")
                self.hint = ui.label().classes("dn-upload-formats")
                self._render = ui.refreshable(self._render_fields)
                self._render()
        self.combine.on_value_change(self.refresh)
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
        if sources and None in self.formulas:
            self.formulas.setdefault(sources[0][0], self.formulas.pop(None))
        if combine:
            formula = self.combined_formula
            if formula is None:
                formula = " + ".join(f"source{i}" for i in range(1, len(sources) + 1))
            field = ui.textarea("Combined transform formula", value=formula).props(
                "dense autogrow rows=2 spellcheck=false"
            ).classes("dn-field dn-formula-input").mark("data-manager-transform-combined-formula")
            field.on_value_change(lambda event: setattr(self, "combined_formula", event.value))
            self.fields.append(field)
        else:
            for index, (source_id, _name) in enumerate(sources or [(None, "")], 1):
                label = (
                    "Source transform formula" if len(sources) <= 1
                    else f"Source {index} transform formula"
                )
                field = ui.textarea(label, value=self.formulas.get(source_id, "source")).props(
                    "dense autogrow rows=2 spellcheck=false"
                ).classes("dn-field dn-formula-input").mark(f"data-manager-transform-formula-{index}")
                field.on_value_change(lambda event, key=source_id: self.formulas.__setitem__(key, event.value))
                self.fields.append(field)

    def set_locked(self, locked):
        for control in (self.enabled, self.combine, *self.fields):
            control.set_enabled(not locked)
