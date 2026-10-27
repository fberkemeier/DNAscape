"""Immutable, user-facing catalogue of the routes exposed by :data:`dnascape.rmap`.

The NiceGUI mapping page needs a little more structure than the scientific
route registry itself: targets which have both an integrated and an ``_xt``
variant should appear once, and a selected route must say exactly which data
fields and shapes it accepts.  This module builds that view from the public
``rmap.targets`` and ``rmap.routes`` APIs.  It deliberately keeps the exact
internal target and method names on every route so the UI never has to infer
how a friendly choice maps back to DNAscape.

The DNAscape import is deferred until :func:`get_mapping_catalog` is first
called.  Importing this lightweight descriptor module therefore does not pull
in the numerical mapping stack during application start-up unless the Mapping
page is actually used.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Literal, TypeAlias, TypeVar, cast

DataForm: TypeAlias = Literal[
    "scalar",
    "x",
    "t",
    "xt",
    "supplied",
    "bundle",
    "regions",
]
InputRole: TypeAlias = Literal["source", "feature_assumption", "regions"]
SettingKind: TypeAlias = Literal[
    "integer",
    "number",
    "boolean",
    "choice",
    "number_pair",
    "number_list",
    "text",
]
SettingCategory: TypeAlias = Literal[
    "domain",
    "fitting",
    "simulation",
    "numerics",
    "integration",
    "advanced",
]
SettingLocation: TypeAlias = Literal["region", "settings", "derived"]
TrackUploadMode: TypeAlias = Literal["unsupported", "single", "series"]
T = TypeVar("T")

DATA_FORMS: tuple[DataForm, ...] = (
    "scalar",
    "x",
    "t",
    "xt",
    "supplied",
    "bundle",
    "regions",
)

DEPENDENCE_ORDER = (
    "constant",
    "space",
    "time",
    "spacetime",
    "regions",
    "ensemble",
)


@dataclass(frozen=True, slots=True)
class RDataFormat:
    """A file format accepted or produced by the public ``rdata`` workflow."""

    key: str
    label: str
    extensions: tuple[str, ...]
    source_supported: bool = True
    target_supported: bool = True
    genomic_track: bool = False
    hint: str = ""


RDATA_FORMATS: tuple[RDataFormat, ...] = (
    RDataFormat("csv", "CSV", (".csv",), hint="Delimited table or numeric profile."),
    RDataFormat("tsv", "TSV", (".tsv",), hint="Tab-delimited table or numeric profile."),
    RDataFormat("txt", "Text", (".txt",), hint="Plain numeric vector or matrix."),
    RDataFormat("npy", "NumPy array", (".npy",), hint="NumPy scalar, vector, or matrix."),
    RDataFormat(
        "npv",
        "NumPy vector",
        (".npv",),
        hint="DNAscape NumPy vector/array (the same binary payload as NPY).",
    ),
    RDataFormat(
        "npz",
        "NumPy archive",
        (".npz",),
        hint="NumPy archive; named arrays and DNAscape form metadata are supported.",
    ),
    RDataFormat(
        "bedgraph",
        "bedGraph",
        (".bedgraph", ".bdg"),
        target_supported=False,
        genomic_track=True,
        hint="Genomic signal track; rdata bins it to a spatial profile.",
    ),
    RDataFormat(
        "bigwig",
        "BigWig",
        (".bigwig", ".bw"),
        target_supported=False,
        genomic_track=True,
        hint="Indexed genomic signal track; rdata bins it to a spatial profile.",
    ),
)

RDATA_SOURCE_FORMAT_KEYS = tuple(item.key for item in RDATA_FORMATS if item.source_supported)
RDATA_TARGET_FORMAT_KEYS = tuple(item.key for item in RDATA_FORMATS if item.target_supported)
RDATA_PLAIN_FORMAT_KEYS = tuple(item.key for item in RDATA_FORMATS if not item.genomic_track)
RDATA_TRACK_FORMAT_KEYS = tuple(item.key for item in RDATA_FORMATS if item.genomic_track)
RDATA_PLAIN_UPLOAD_ACCEPT = (
    ",".join(
        extension
        for item in RDATA_FORMATS
        if item.source_supported and not item.genomic_track
        for extension in item.extensions
    )
    + ",.gz"
)
RDATA_TRACK_UPLOAD_ACCEPT = ",".join(
    extension
    for item in RDATA_FORMATS
    if item.source_supported and item.genomic_track
    for extension in item.extensions
)
RDATA_UPLOAD_ACCEPT = (
    ",".join(
        extension
        for item in RDATA_FORMATS
        if item.source_supported
        for extension in item.extensions
    )
    + ",.gz"
)
RDATA_UPLOAD_HINT = (
    "CSV, TSV, TXT, NPY/NPV, or NPZ; spatial inputs also accept bedGraph and BigWig. "
    "Gzip-compressed plain-text and bedGraph files are supported."
)
RDATA_FORM_HINT = (
    "DNAscape reads constants as scalars, spatial profiles as one-dimensional "
    "arrays, temporal profiles as row arrays (1 x time), and space-time fields "
    "as position-by-time matrices."
)


@dataclass(frozen=True, slots=True)
class MappingInput:
    """One scientific data input requested by an exact mapping route."""

    key: str
    label: str
    role: InputRole
    forms: tuple[DataForm, ...]
    required: bool
    default: Any
    default_provided: bool
    description: str
    accepted_formats: tuple[str, ...]
    accept: str
    form_formats: tuple[tuple[DataForm, tuple[str, ...]], ...]
    form_accepts: tuple[tuple[DataForm, str], ...]
    form_multiples: tuple[tuple[DataForm, bool], ...]
    form_track_modes: tuple[tuple[DataForm, TrackUploadMode], ...]
    hint: str
    prepared_result_only: bool = False

    @property
    def has_default(self) -> bool:
        """Whether rmap supplies this data-like assumption when it is omitted."""

        return self.default_provided

    @property
    def allows_constant(self) -> bool:
        """Whether the value may be entered as a scalar rather than uploaded."""

        return "scalar" in self.forms

    @property
    def allows_upload(self) -> bool:
        """Whether an ordinary rdata-supported upload can satisfy the input."""

        return bool(self.accepted_formats) and not self.prepared_result_only

    def formats_for(self, form: DataForm) -> tuple[str, ...]:
        """Return the exact rdata format keys suitable for one selected form."""

        for candidate, formats in self.form_formats:
            if candidate == form:
                return formats
        raise KeyError(f"Input {self.key!r} does not accept form {form!r}.")

    def accept_for(self, form: DataForm) -> str:
        """Return the browser upload ``accept`` value for one selected form."""

        for candidate, accept in self.form_accepts:
            if candidate == form:
                return accept
        raise KeyError(f"Input {self.key!r} does not accept form {form!r}.")

    def multiple_for(self, form: DataForm) -> bool:
        """Whether the upload control should accept several files for a form.

        A space-time field can be supplied either as one ordinary matrix or as
        several genomic tracks (one track per time point).  Enabling multiple
        selection does not make multiple files mandatory for matrix formats.
        """

        for candidate, multiple in self.form_multiples:
            if candidate == form:
                return multiple
        raise KeyError(f"Input {self.key!r} does not accept form {form!r}.")

    def track_mode_for(self, form: DataForm) -> TrackUploadMode:
        """Return whether genomic tracks are unsupported, singular, or a series."""

        for candidate, mode in self.form_track_modes:
            if candidate == form:
                return mode
        raise KeyError(f"Input {self.key!r} does not accept form {form!r}.")


@dataclass(frozen=True, slots=True)
class MappingSetting:
    """One user-facing, route-specific rmap assumption.

    ``has_default`` is separate from ``default`` because ``None`` is a real
    package default for automatic axes, bounds, and early-stop thresholds.
    ``location`` lets the page keep domain controls in its Region section and
    omit values derived from those controls without duplicating them in the
    Mapping settings section.
    """

    key: str
    label: str
    category: SettingCategory
    location: SettingLocation
    kind: SettingKind
    required: bool
    has_default: bool
    default: Any
    allowed_values: tuple[Any, ...]
    description: str
    unit: str | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    nullable: bool = True


@dataclass(frozen=True, slots=True)
class MappingRoute:
    """An exact implemented rmap method and all metadata needed to configure it."""

    target: str
    target_key: str
    method: str
    label: str
    description: str
    inputs: tuple[MappingInput, ...]
    settings: tuple[MappingSetting, ...]
    output_form: DataForm
    output_dependence: str
    target_shapes: tuple[str, ...]
    default_target_shape: str | None
    required_assumptions: tuple[str, ...]
    optional_assumptions: tuple[str, ...]
    defaults: tuple[tuple[str, Any], ...]
    assumption_forms: tuple[tuple[str, tuple[DataForm, ...]], ...]
    assumption_values: tuple[tuple[str, tuple[Any, ...]], ...]
    prepared_result_only: bool = False

    @property
    def source_keys(self) -> tuple[str, ...]:
        """Required rmap source and feature-assumption names, in call order."""

        return tuple(item.key for item in self.inputs)

    @property
    def defaults_dict(self) -> dict[str, Any]:
        """Return a fresh mutable mapping for passing to ``rmap``."""

        return dict(self.defaults)

    def input(self, key: str) -> MappingInput:
        """Look up one route input by its exact rmap name."""

        for item in self.inputs:
            if item.key == key:
                return item
        raise KeyError(f"Route {self.method!r} has no input {key!r}.")

    def setting(self, key: str) -> MappingSetting:
        """Look up one configurable or derived route assumption."""

        for item in self.settings:
            if item.key == key:
                return item
        raise KeyError(f"Route {self.method!r} has no setting {key!r}.")

    def settings_for(
        self,
        category: SettingCategory | None = None,
        *,
        location: SettingLocation | None = None,
    ) -> tuple[MappingSetting, ...]:
        """Filter settings by semantic category and/or UI location."""

        return tuple(
            item
            for item in self.settings
            if (category is None or item.category == category)
            and (location is None or item.location == location)
        )

    def active_settings(
        self, target_shape: str | None = None, input_forms: Mapping[str, DataForm] | None = None,
    ) -> tuple[MappingSetting, ...]:
        """Settings consumed by this method for the selected target shape.

        The registry also documents compatibility options accepted by broad
        wrappers. A setting which the active branch ignores must not become
        an editable control or survive a change of target shape.
        """

        shape = self.default_target_shape if target_shape is None else target_shape
        settings = tuple(item for item in self.settings if item.key not in self.single_phase_fit_aliases)
        if self.target == "initiation_rate" and self.target_shapes:
            if shape == "weibull":
                settings = tuple(item for item in settings if item.key in _WEIBULL_FIRING_SETTINGS)
                if self.output_dependence == "time":
                    settings = tuple(item for item in settings if item.key != "alpha_smooth")
            else:
                settings = tuple(
                    item for item in settings if item.key not in _WEIBULL_ONLY_SETTINGS
                    | {"spacetime_space_fit_step", "spacetime_time_fit_step"}
                )
                settings = tuple(
                    replace(item, allowed_values=("space_then_time", "time_then_space"))
                    if item.key == "spacetime_order"
                    else item
                    for item in settings
                )
        elif self.target == "fork_speed" and any(item.key == "tolerance" for item in settings):
            unused = set()
            if self.output_dependence not in {"space", "spacetime"}:
                unused.add("spatial_knots")
            if self.output_dependence not in {"time", "spacetime"}:
                unused.add("temporal_knots")
            if self.output_dependence == "constant":
                unused.add("regularization")
            settings = tuple(item for item in settings if item.key not in unused)
        elif self.target == "fork_speed" and self.output_dependence == "time" and shape == "best":
            settings = tuple(item for item in settings if item.key in _TEMPORAL_SPEED_SETTINGS)
        elif self.method == "timing_and_firing_to_constant_fork_speed":
            settings = tuple(item for item in settings if item.key != "neighbourhood_range")
        elif (
            self.method
            in {
                "timing_and_time_firing_to_constant_fork_speed",
                "timing_and_time_firing_to_spatial_fork_speed",
            }
            and shape == "best"
        ):
            settings = tuple(
                item for item in settings if item.key not in {"sim_number", "refine_seed"}
            )
        elif self.method.startswith("firing_and_fraction_to_initiation_pdf"):
            settings = tuple(item for item in settings if item.key in _PDF_SETTINGS)
        elif self.method == "fork_density_profiles_to_directionality":
            settings = ()  # Already time-integrated profiles need no integration axis.
        elif self.target == "origin_efficiency" and self.output_dependence == "regions":
            if all("xt" not in item.forms for item in self.inputs):
                settings = tuple(
                    item for item in settings if item.key not in {"time_grid", "resolution_time"}
                )
        if not self.supports_periodic(shape):
            settings = tuple(item for item in settings if item.key != "perQ")
        if self.output_dependence in {"time", "spacetime"} and (
            (self.target == "fork_speed" and not any(item.key == "tolerance" for item in settings))
            or (self.target == "initiation_rate" and shape == "free")
        ):
            # These wrapper aliases are redundant with the actual phase
            # controls and can be shadowed by their visible default values.
            settings = tuple(
                item for item in settings if item.key not in {"fit_step", "maxiter", "sim_number"}
            )
        if any("bundle" in item.forms for item in self.inputs):
            # Prepared results supply intrinsic axes. Retain descriptors for
            # service validation and metadata promotion without rendering
            # controls which could contradict the stored coordinates.
            settings = tuple(
                replace(item, location="derived")
                if item.key in {"resolution", "resolution_time", "time_grid", "x_array"}
                else item
                for item in settings
            )
        if (
            input_forms is not None and self.target == "initiation_rate"
            and self.output_dependence in {"constant", "space"}
            and input_forms.get("fork_speed") not in {"t", "xt"}
        ):
            settings = tuple(item for item in settings if item.key not in {"time_grid", "resolution_time"})
        from .mapping_defaults import mapping_defaults

        return mapping_defaults(self, settings, shape)

    @property
    def single_phase_fit_aliases(self) -> dict[str, str]:
        """Compatibility aliases for a route with just one spatial fit phase."""

        if self.output_dependence not in {"constant", "space"}:
            return {}
        keys = {item.key for item in self.settings}
        return {
            alias: canonical
            for alias, canonical in (("space_maxiter", "maxiter"), ("space_fit_step", "fit_step"))
            if {alias, canonical} <= keys
        }

    def supports_periodic(self, target_shape: str | None = None) -> bool:
        """Whether the scientific branch can execute periodic boundaries."""

        return True

    def shapes_for(self, input_forms: Mapping[str, DataForm]) -> tuple[str, ...]:
        """Target shapes implemented for the currently supplied data forms."""

        return self.target_shapes

    @property
    def user_settings(self) -> tuple[MappingSetting, ...]:
        """Settings which should have a direct UI control."""

        return tuple(item for item in self.settings if item.location != "derived")

    @property
    def setting_defaults_dict(self) -> dict[str, Any]:
        """Return defaults for route settings, including explicit ``None``."""

        return {item.key: item.default for item in self.settings if item.has_default}


@dataclass(frozen=True, slots=True)
class MappingRouteFamily:
    """One user-facing combination of biological inputs.

    ``rmap`` registers an exact method for each supported input-shape and
    algorithm combination.  Those methods are deliberately retained in
    :attr:`variants`, but exposing every one in a route dropdown makes the
    user choose implementation details before choosing their data.  A family
    instead groups exact methods which use the same factors for one target and
    output dependence.  Input-form choices progressively constrain the exact
    variants that remain valid.
    """

    key: str
    target: str
    target_key: str
    label: str
    description: str
    inputs: tuple[MappingInput, ...]
    required_factor_keys: tuple[str, ...]
    variants: tuple[MappingRoute, ...]
    output_form: DataForm
    output_dependence: str
    target_shapes: tuple[str, ...]
    default_target_shape: str | None
    prepared_result_only: bool = False

    @property
    def methods(self) -> tuple[str, ...]:
        """Exact implemented ``rmap`` methods represented by this family."""

        return tuple(route.method for route in self.variants)

    @property
    def factor_keys(self) -> tuple[str, ...]:
        """Required biological factors which define the route label."""

        return self.required_factor_keys

    @property
    def input_keys(self) -> tuple[str, ...]:
        """All displayed inputs, including optional defaulted overrides."""

        return tuple(item.key for item in self.inputs)

    @property
    def optional_input_keys(self) -> tuple[str, ...]:
        """Defaulted feature inputs which may be overridden by the user."""

        return tuple(key for key in self.input_keys if key not in self.required_factor_keys)

    def input(self, key: str) -> MappingInput:
        """Look up a merged family input descriptor."""

        for item in self.inputs:
            if item.key == key:
                return item
        raise KeyError(f"Route family {self.key!r} has no input {key!r}.")

    def variant_input(self, route: MappingRoute, key: str) -> MappingInput:
        """Return the exact descriptor backing a biological family input.

        Integrated and space-time registry inputs sometimes use different
        parameter names (for example ``initiation_events`` and
        ``initiation_events_xt``).  The family deliberately presents them as
        one factor whose selected form identifies the exact parameter.
        """

        if all(route.method != variant.method for variant in self.variants):
            raise ValueError(f"Route {route.method!r} is not a variant of {self.key!r}.")
        matches = tuple(item for item in route.inputs if _friendly_key(item.key) == key)
        if len(matches) != 1:
            raise KeyError(f"Route {route.method!r} has no unambiguous family input {key!r}.")
        return matches[0]

    def exact_input_key(self, route: MappingRoute, key: str) -> str:
        """Translate a family factor key to the resolved method parameter."""

        return self.variant_input(route, key).key

    def compatible_routes(
        self,
        selections: Mapping[str, DataForm] | None = None,
    ) -> tuple[MappingRoute, ...]:
        """Return exact variants compatible with a partial form selection.

        Unknown input names and unknown forms intentionally produce no
        candidates.  This makes stale browser state fail closed when changing
        targets or route families.
        """

        selected = {} if selections is None else dict(selections)
        if any(
            key not in self.input_keys or form not in DATA_FORMS for key, form in selected.items()
        ):
            return ()
        return tuple(
            route
            for route in self.variants
            if all(
                any(_friendly_key(item.key) == key and form in item.forms for item in route.inputs)
                for key, form in selected.items()
            )
        )

    def allowed_forms(
        self,
        input_key: str,
        selections: Mapping[str, DataForm] | None = None,
    ) -> tuple[DataForm, ...]:
        """Return forms allowed for one input given selections on the others.

        The current value of ``input_key`` is ignored deliberately.  A UI can
        therefore ask for all alternatives for an already-selected dropdown
        while the other inputs continue to constrain it.
        """

        self.input(input_key)  # Validate the requested key even for no matches.
        other_selections: dict[str, DataForm] = {
            key: form
            for key, form in (() if selections is None else selections.items())
            if key != input_key
        }
        compatible = self.compatible_routes(other_selections)
        return tuple(
            form
            for form in DATA_FORMS
            if any(
                form in item.forms
                for route in compatible
                for item in route.inputs
                if _friendly_key(item.key) == input_key
            )
        )

    def resolve(self, selections: Mapping[str, DataForm]) -> MappingRoute:
        """Resolve complete input-form choices to one exact implemented route.

        Some registry methods overlap intentionally (for example a narrow
        deterministic mapping and a general simulation both accept constant
        kinetic inputs).  Resolution follows the registry's stable order,
        matching ``rmap``'s own first-compatible-route behaviour.
        """

        selected = dict(selections)
        missing = tuple(key for key in self.required_factor_keys if key not in selected)
        extras = tuple(key for key in selected if key not in self.input_keys)
        if missing or extras:
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extras:
                details.append(f"unknown: {', '.join(extras)}")
            raise ValueError(f"Incomplete route-form selection ({'; '.join(details)}).")

        compatible = self.compatible_routes(selected)
        if not compatible:
            forms = ", ".join(f"{key}={value}" for key, value in selected.items())
            raise ValueError(f"No implemented rmap method accepts {forms}.")

        # rmap resolves overlapping exact specifications by taking the first
        # accepted registry candidate.  Families preserve that registry order,
        # so the application makes the same deterministic choice.
        return compatible[0]


@dataclass(frozen=True, slots=True)
class TargetVariant:
    """One output-dependence choice within a friendly target."""

    internal_target: str
    dependence: str
    label: str
    output_form: DataForm
    target_shapes: tuple[str, ...]
    default_target_shape: str | None
    description: str
    methods: tuple[str, ...]
    prepared_result_only: bool = False


@dataclass(frozen=True, slots=True)
class MappingTarget:
    """A friendly target choice, possibly merging an integrated and ``_xt`` pair."""

    key: str
    label: str
    description: str
    variants: tuple[TargetVariant, ...]
    default_dependence: str
    prepared_result_only: bool = False
    requires_simulation: bool = False

    @property
    def dependences(self) -> tuple[str, ...]:
        return tuple(item.dependence for item in self.variants)

    @property
    def internal_targets(self) -> tuple[str, ...]:
        return _unique(item.internal_target for item in self.variants)

    def variant(self, dependence: str | None = None) -> TargetVariant:
        """Return a dependence variant, using the target default when omitted."""

        wanted = self.default_dependence if dependence is None else dependence
        for item in self.variants:
            if item.dependence == wanted:
                return item
        raise KeyError(f"Target {self.key!r} has no {wanted!r} output dependence.")


@dataclass(frozen=True, slots=True)
class MappingCatalog:
    """Immutable snapshot of implemented rmap conversions exposed by Mapping."""

    targets: tuple[MappingTarget, ...]
    routes: tuple[MappingRoute, ...]
    route_families: tuple[MappingRouteFamily, ...]
    public_target_ids: tuple[str, ...]
    formats: tuple[RDataFormat, ...] = RDATA_FORMATS

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def internal_target_count(self) -> int:
        return len(self.public_target_ids)

    @property
    def route_count(self) -> int:
        return len(self.routes)

    @property
    def route_family_count(self) -> int:
        return len(self.route_families)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(route.method for route in self.routes)

    def target(self, key: str) -> MappingTarget:
        """Look up a friendly target key or any exact internal target name."""

        from dnascape._terminology import canonical_rate_key
        key = canonical_rate_key(key)
        for target in self.targets:
            if target.key == key or key in target.internal_targets:
                return target
        raise KeyError(f"Unknown mapping target {key!r}.")

    def route(self, method: str) -> MappingRoute:
        """Look up an exact public rmap method."""

        from dnascape._terminology import canonical_rate_key
        method = canonical_rate_key(method)
        for route in self.routes:
            if route.method == method:
                return route
        raise KeyError(f"Unknown mapping method {method!r}.")

    def route_family(self, key: str) -> MappingRouteFamily:
        """Look up one user-facing factor-combination family."""

        for family in self.route_families:
            if family.key == key:
                return family
        raise KeyError(f"Unknown mapping route family {key!r}.")

    def routes_for(
        self,
        target: str | None = None,
        dependence: str | None = None,
        *,
        internal_target: str | None = None,
    ) -> tuple[MappingRoute, ...]:
        """Filter routes for a friendly target, dependence, or exact target.

        A value such as ``"replicated_fraction"`` is treated as the friendly
        target and therefore includes both its time and space-time variants.
        Pass ``internal_target=`` when an exact registry target is required.
        """

        target_key = self.target(target).key if target is not None else None
        return tuple(
            route
            for route in self.routes
            if (target_key is None or route.target_key == target_key)
            and (dependence is None or route.output_dependence == dependence)
            and (internal_target is None or route.target == internal_target)
        )

    def route_families_for(
        self,
        target: str | None = None,
        dependence: str | None = None,
        *,
        internal_target: str | None = None,
    ) -> tuple[MappingRouteFamily, ...]:
        """Filter factor-combination families by target and dependence."""

        target_key = self.target(target).key if target is not None else None
        return tuple(
            family
            for family in self.route_families
            if (target_key is None or family.target_key == target_key)
            and (dependence is None or family.output_dependence == dependence)
            and (internal_target is None or family.target == internal_target)
        )

    def signature(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return a compact deterministic signature for scientific drift tests."""

        return self.public_target_ids, self.methods


_FEATURE_LABELS = {
    "initiation_rate": "Initiation rate",
    "fork_speed": "Fork speed",
    "replication_timing": "Replication timing",
    "fork_directionality": "Fork directionality",
    "replicated_fraction": "Replicated fraction",
    "replicated_fraction_xt": "Replicated fraction",
    "fork_density_xt": "Total fork density",
    "fork_densities": "Directional fork densities",
    "fork_densities_xt": "Directional fork densities",
    "right_fork_density": "Right-moving fork density",
    "right_fork_density_xt": "Right-moving fork density",
    "left_fork_density": "Left-moving fork density",
    "left_fork_density_xt": "Left-moving fork density",
    "origin_efficiency": "Origin efficiency",
    "inter_origin_distance": "Inter-origin distance",
    "s_phase_duration": "S-phase duration",
    "completion_time": "Completion time",
    "num_origins": "Number of origins",
    "initiation_events": "Initiation events",
    "initiation_events_xt": "Initiation events",
    "initiation_pdf": "Initiation probability density",
    "initiation_pdf_xt": "Initiation probability density",
    "coalescence_events": "Coalescence events",
    "coalescence_events_xt": "Coalescence events",
    "coalescence_pdf": "Coalescence probability density",
    "coalescence_pdf_xt": "Coalescence probability density",
    "regions": "Regions",
}

_FEATURE_DESCRIPTIONS = {
    "initiation_rate": "Initiation rate per spatial bin (min^-1), supplied as a constant or kinetic field.",
    "fork_speed": "Replication-fork progression speed in kb/min.",
    "replication_timing": "Replication timing as a profile along genomic position.",
    "fork_directionality": "Net fork movement directionality along genomic position.",
    "replicated_fraction_xt": "Prepared position-by-time replicated-fraction field.",
    "fork_densities_xt": (
        "Prepared DNAscape result containing paired right- and left-moving "
        "position-by-time fork-density fields."
    ),
    "right_fork_density": "Time-integrated right-moving fork-density profile.",
    "left_fork_density": "Time-integrated left-moving fork-density profile.",
    "right_fork_density_xt": "Position-by-time right-moving fork-density field.",
    "left_fork_density_xt": "Position-by-time left-moving fork-density field.",
    "initiation_events": "Expected initiations per kb, integrated over the time window.",
    "initiation_events_xt": "Expected initiations per kb per minute: I(x,t) [1-f(x,t)].",
    "initiation_pdf": "Time-integrated initiation probability-density profile.",
    "initiation_pdf_xt": "Position-by-time initiation probability-density field.",
    "coalescence_events": "Expected fork coalescences per kb, integrated over the time window.",
    "coalescence_events_xt": "Expected coalescences per kb per minute: 2 v rho+ rho- / (1-f).",
    "coalescence_pdf_xt": "Position-by-time coalescence probability-density field.",
    "s_phase_duration": (
        "Elapsed time from the first realised initiation to replication completion."
    ),
    "regions": "One (start, end) pair, a list of pairs, or a name-to-pair mapping.",
}

_TARGET_GROUP_DESCRIPTIONS = {
    "replicated_fraction": "Replicated fraction over time or across position and time.",
    "initiation_events": "Expected initiation density, or initiation rate density resolved over time.",
    "initiation_pdf": "Initiation probability density as a spatial profile or space-time field.",
    "coalescence_events": "Expected coalescence density, or coalescence rate density resolved over time.",
    "coalescence_pdf": "Coalescence probability density as a spatial profile or space-time field.",
    "origin_efficiency": (
        "Origin efficiency across genomic position or integrated over user-defined regions."
    ),
}

_SETTING_DESCRIPTIONS = {
    "resolution": "Spatial resolution along the genome.",
    "resolution_time": "Temporal resolution used for time-dependent fields.",
    "time_grid": "Explicit time axis; leave automatic to derive it from the time controls.",
    "T0": "Reference start time for converting timing values to relative time.",
    "max_time": "End of the computed time domain when no input time axis is supplied.",
    "max_rep_time": "Maximum replication time allowed by the selected route.",
    "n_bins": "Number of spatial bins, derived from the selected region when omitted.",
    "genome_length": "Genome or interval length used to infer the number of bins.",
    "interval_kb": "Genomic interval used to infer or select the mapped region.",
    "perQ": "Treat the region as periodic rather than linear.",
    "geometry": "Genome boundary geometry used by stochastic simulation.",
    "x_array": "Explicit spatial coordinates for profile integration.",
    "neighbourhood_range": "Spatial neighbourhood used by deterministic mappings.",
    "fit_step": "Global multiplicative fitting update size.",
    "maxiter": "Maximum number of global fitting iterations.",
    "error": "Optional MSE threshold for early stopping; automatic disables it.",
    "sim_number": "Number of stochastic simulations in the active route or fit phase.",
    "err_threshold": "Numerical floor used while fitting initiation-rate values.",
    "deviation_threshold": "Tolerance below which a spatial update is left unchanged.",
    "rate_bounds": "Lower and upper bounds for initiation-rate fitting.",
    "speed_bounds": "Lower and upper bounds for fork-speed fitting.",
    "speed_regularization": (
        "Noise-adaptive smoothness penalty on spatial fork speeds in the integrated "
        "timing fit using supplied directionality; zero disables the penalty."
    ),
    "firing_bounds": "Lower and upper bounds for fitted initiation rates.",
    "initial_fork_speed": "Starting fork-speed value for fitting, in kb/min.",
    "initial_initiation_rate": "Starting initiation-rate value; automatic uses the route's native estimate.",
    "space_fit_step": "Spatial-phase fitting update size.",
    "space_maxiter": "Maximum number of spatial-phase fitting iterations.",
    "temporal_smoothw": "Optional smoothing window for time-dependent profiles.",
    "temporal_fit_step": "Temporal-phase fitting update size.",
    "temporal_maxiter": "Maximum number of temporal-phase fitting iterations.",
    "temporal_sim_number": "Simulations per temporal refinement iteration.",
    "spacetime_fit_step": "Local space-time update size for mixed fitting.",
    "spacetime_space_fit_step": "Spatial update size for factorized space-time fitting.",
    "spacetime_time_fit_step": "Temporal update size for factorized space-time fitting.",
    "spacetime_sim_number": "Simulations per mixed space-time refinement iteration.",
    "spacetime_update": "Use factorized space/time phases or mixed local corrections.",
    "spacetime_order": "Order of the spatial and temporal phases in a factorized fit.",
    "spacetime_kernel_t": "Temporal kernel width for mixed space-time corrections.",
    "weibull_shape": "Weibull shape parameter; 1 is exponential and 2 is linear hazard.",
    "weibull_delay": "Activation delay for a Weibull initiation-rate target.",
    "fit_weibull_shape": "Fit the Weibull shape as well as the rate field.",
    "alpha_smooth": "Smoothness regularisation applied to log(alpha).",
    "alpha_prior": "Prior strength keeping log(alpha) near its initial estimate.",
    "floor": "Lower numerical floor for fitted initiation-rate fields.",
    "unreplicated_floor": "Numerical floor when conditioning on unreplicated DNA.",
    "smoothw": "Spatial smoothing window for fitted profiles.",
    "refine_ratio_bounds": "Lower and upper bounds for timing-ratio refinement updates.",
    "refine_edge_kb": (
        "Maximum non-periodic endpoint context used to stabilize the forward fit. "
        "DNAscape caps this adaptively and tapers only the very last bins; set it "
        "to zero when the supplied endpoints are true chromosome ends."
    ),
    "refine_seed": "Random-seed offset for simulation-refined fitting.",
    "refine_verbose": "Show progress from refinement sub-loops.",
    "stall_rate": "Legacy probability of a stall while crossing one spatial bin.",
    "stall_rate_per_kb": (
        "Physical stall density. DNAscape converts it with the selected spatial "
        "resolution using p = 1 - exp(-rate x resolution)."
    ),
    "tau": "Fork-stall lifetime; infinity disables recovery.",
    "cfl": "Maximum Courant factor used for mean-field integration substeps.",
    "seed": "Optional random seed for reproducible simulation routes.",
    "ffiring_nr": "Finite firing-factor pool size; automatic disables the limitation.",
    "ffiring_recycle": "Rate at which used firing factors return to the available pool.",
    "ffiring_forkQ": "Include active forks in the finite firing-factor pool.",
    "region_names": "Optional labels matching the supplied integration regions.",
}

_SETTING_LABELS = {
    "error": "MSE threshold",
    "err_threshold": "Initiation rate floor",
    "spatial_knots": "Spatial control points",
    "temporal_knots": "Temporal control points",
    "regularization": "Smoothness penalty",
    "tolerance": "Convergence tolerance",
    "tail_policy": "Unobserved completion tail",
    "initial_policy": "Initial replication state",
    "invalid_policy": "Unidentifiable values",
    "T0": "Timing anchor T0",
    "perQ": "Periodic",
    "cfl": "CFL factor",
    "sim_number": "Simulations",
    "tau": "Recovery time",
    "maxiter": "Fit iterations",
    "n_bins": "Spatial bins",
    "x_array": "Position axis",
    "ffiring_nr": "Firing-factor pool",
    "ffiring_recycle": "Firing-factor recycling",
    "ffiring_forkQ": "Fork-limited firing factors",
    "stall_rate_per_kb": "Stall rate",
    "refine_edge_kb": "Maximum edge context",
    "speed_regularization": "Speed regularization",
}

_SETTING_DESCRIPTIONS.update({
    "spatial_knots": "Number of spatial control points in the positive interpolated kinetic field; capped by available bins.",
    "temporal_knots": "Number of temporal control points; more points permit variation that timing alone may not constrain.",
    "regularization": "Penalty on differences between neighboring log-kinetic control points, relative to normalized timing error.",
    "tolerance": "Relative objective and step convergence tolerance of the bounded analytic-gradient fitter.",
    "tail_policy": "Require observed completion, report a restricted mean at the last sample, or extrapolate an exponential survival tail.",
    "initial_policy": "Require an initially unreplicated sample or condition timing on the DNA still unreplicated at the first sample.",
    "tail_fit_points": "Number of terminal samples used only for exponential-tail extrapolation.",
    "tail_tolerance": "Largest surviving fraction treated as complete.",
    "invalid_policy": "Mask unidentifiable ratios with NaN, explicitly fill them with zero, or stop with an error.",
    "density_floor": "Fork density below this threshold cannot identify speed from the replication-fraction derivative.",
    "monotonic_policy": "Reject decreasing fractions or project noisy samples onto their cumulative upper envelope.",
    "temporal_smoothw": "Time samples in the moving-average smoothing window before differentiating the replicated fraction.",
    "spatial_average": "Average equally over supplied bins or weight by their spatial coordinates.",
})

_SETTING_CATEGORIES: dict[str, SettingCategory] = {
    **{
        name: "domain"
        for name in (
            "resolution",
            "resolution_time",
            "time_grid",
            "T0",
            "max_time",
            "max_rep_time",
            "n_bins",
            "genome_length",
            "interval_kb",
            "perQ",
            "geometry",
            "x_array",
        )
    },
    **{
        name: "simulation"
        for name in (
            "sim_number",
            "seed",
            "stall_rate_per_kb",
            "tau",
            "ffiring_nr",
            "ffiring_recycle",
            "ffiring_forkQ",
        )
    },
    **{
        name: "numerics"
        for name in ("neighbourhood_range", "cfl", "floor", "firing_floor", "unreplicated_floor")
    },
    "region_names": "integration",
}

_SETTING_LOCATIONS: dict[str, SettingLocation] = {
    **{
        name: "region"
        for name in (
            "resolution",
            "resolution_time",
            "time_grid",
            "T0",
            "max_time",
            "max_rep_time",
            "perQ",
            "geometry",
        )
    },
    **{name: "derived" for name in ("n_bins", "genome_length", "interval_kb", "x_array", "max_time", "max_rep_time")},
}

_SETTING_CHOICES: dict[str, tuple[Any, ...]] = {
    "geometry": ("linear", "periodic"),
    "spacetime_update": ("factorized", "mixed"),
    "spacetime_order": ("crossed", "mixed", "space_then_time", "time_then_space"),
}

_BOOLEAN_SETTINGS = {
    "perQ",
    "fit_weibull_shape",
    "refine_verbose",
    "ffiring_forkQ",
}
_INTEGER_SETTINGS = {
    "tail_fit_points",
    "temporal_smoothw",
    "spatial_knots",
    "temporal_knots",
    "smoothing_window",
    "maxiter",
    "sim_number",
    "space_maxiter",
    "temporal_maxiter",
    "temporal_sim_number",
    "spacetime_sim_number",
    "seed",
    "refine_seed",
    "n_bins",
    "ffiring_nr",
}
_PAIR_SETTINGS = {
    "rate_bounds",
    "speed_bounds",
    "firing_bounds",
    "refine_ratio_bounds",
    "interval_kb",
}
_LIST_SETTINGS = {"time_grid", "x_array"}
_TEXT_SETTINGS = {"region_names"}
_POSITIVE_SETTINGS = {
    "tolerance",
    "resolution",
    "resolution_time",
    "fit_step",
    "space_fit_step",
    "temporal_fit_step",
    "spacetime_fit_step",
    "spacetime_space_fit_step",
    "spacetime_time_fit_step",
    "weibull_shape",
    "spacetime_kernel_t",
    "initial_fork_speed",
    "initial_initiation_rate",
    "genome_length",
    "tau",
}
_NONNEGATIVE_SETTINGS = {
    "speed_regularization",
    "tail_tolerance",
    "density_floor",
    "regularization",
    "T0",
    "max_time",
    "max_rep_time",
    "neighbourhood_range",
    "error",
    "err_threshold",
    "deviation_threshold",
    "weibull_delay",
    "alpha_smooth",
    "alpha_prior",
    "floor",
    "unreplicated_floor",
    "smoothw",
    "temporal_smoothw",
    "refine_edge_kb",
    "refine_seed",
    "stall_rate",
    "stall_rate_per_kb",
    "ffiring_recycle",
}
_SETTING_UNITS = {
    "resolution": "kb/bin",
    "resolution_time": "min/bin",
    "T0": "min",
    "max_time": "min",
    "max_rep_time": "min",
    "genome_length": "kb",
    "interval_kb": "kb",
    "neighbourhood_range": "kb",
    "initial_fork_speed": "kb/min",
    "refine_edge_kb": "kb",
    "weibull_delay": "min",
    "stall_rate_per_kb": "stalls/kb",
    "tau": "min",
}

# These are accepted by broad wrappers but are either function-call plumbing,
# diagnostics, memory tuning, or output-retention switches which the selected
# mapping target determines.  They must not become misleading UI controls.
_HIDDEN_ASSUMPTIONS = {
    "source",
    "assumptions",
    "target_form",
    "target_shape",
    "verbose",
    "print_every",
    "dens_block_n",
    "dens_memmap_dir",
    "time_statsQ",
    "time_stats_xtQ",
    "time_stats_densQ",
    "retain_rep_times_per_sim",
    "retain_rep_times_limit",
    "retain_inter_origin_distances",
    "retain_inter_origin_distances_limit",
    "stall_rate",
}

_WEIBULL_ONLY_SETTINGS = {
    "weibull_shape",
    "weibull_delay",
    "fit_weibull_shape",
    "alpha_smooth",
    "alpha_prior",
}
_WEIBULL_FIRING_SETTINGS = _WEIBULL_ONLY_SETTINGS | {
    "rate_bounds", "initial_initiation_rate",
    "perQ",
    "stall_rate_per_kb",
    "tau",
    "sim_number",
    "refine_seed",
    "resolution",
    "resolution_time",
    "time_grid",
    "T0",
    "maxiter",
    "error",
    "refine_edge_kb",
    "refine_verbose",
    "floor",
}
_TEMPORAL_SPEED_SETTINGS = {
    "perQ",
    "resolution",
    "resolution_time",
    "time_grid",
    "T0",
    "fit_step",
    "maxiter",
    "error",
    "temporal_fit_step",
    "temporal_maxiter",
    "speed_bounds",
    "initial_fork_speed",
    "refine_edge_kb",
    "refine_verbose",
    "n_bins",
}
_PDF_SETTINGS = {"resolution", "resolution_time", "time_grid", "x_array"}

_SIGNATURE_ALIASES = {
    "resolution_space": "resolution",
    "resolution_x": "resolution",
    "resolution_t": "resolution_time",
}

_SETTING_FALLBACK_DEFAULTS: dict[str, Any] = {
    "resolution": 1.0,
    "resolution_time": 5.0,
    "time_grid": None,
    "max_time": None,
    "max_rep_time": None,
    "n_bins": None,
    "genome_length": None,
    "interval_kb": None,
    "perQ": False,
    "error": None,
    "maxiter": 10,
    "weibull_shape": 2.0,
    "weibull_delay": 0.0,
    "fit_weibull_shape": False,
    "alpha_smooth": 1e-2,
    "alpha_prior": 0.0,
    "space_maxiter": 10,
    "temporal_fit_step": 0.35,
    "temporal_maxiter": 10,
    "temporal_sim_number": 100,
    "spacetime_fit_step": 0.25,
    "spacetime_sim_number": 100,
    "spacetime_update": "factorized",
    "spacetime_order": "crossed",
    "spacetime_kernel_t": 1.0,
    "sim_number": 100,
    "stall_rate_per_kb": 0.0,
}

_FIXED_OUTPUTS: dict[str, tuple[DataForm, str]] = {
    "replication_timing": ("x", "space"),
    "replicated_fraction": ("t", "time"),
    "replicated_fraction_xt": ("xt", "spacetime"),
    "fork_directionality": ("x", "space"),
    "fork_densities_xt": ("bundle", "spacetime"),
    "right_fork_density_xt": ("xt", "spacetime"),
    "left_fork_density_xt": ("xt", "spacetime"),
    "initiation_events": ("x", "space"),
    "initiation_events_xt": ("xt", "spacetime"),
    "initiation_pdf": ("x", "space"),
    "initiation_pdf_xt": ("xt", "spacetime"),
    "coalescence_events": ("x", "space"),
    "coalescence_events_xt": ("xt", "spacetime"),
    "coalescence_pdf": ("x", "space"),
    "coalescence_pdf_xt": ("xt", "spacetime"),
    "inter_origin_distance": ("supplied", "ensemble"),
    "s_phase_duration": ("supplied", "ensemble"),
    "completion_time": ("supplied", "ensemble"),
    "num_origins": ("supplied", "ensemble"),
}

_DEFAULT_DEPENDENCE = {
    "initiation_rate": "space",
    "fork_speed": "space",
    "replicated_fraction": "time",
    "initiation_events": "space",
    "initiation_pdf": "space",
    "coalescence_events": "space",
    "coalescence_pdf": "space",
    "origin_efficiency": "space",
}

_DEPENDENCE_LABELS = {
    "constant": "Constant",
    "space": "Space",
    "time": "Time",
    "spacetime": "Space-time",
    "regions": "Regions",
    "ensemble": "Ensemble",
}

_EVENT_DENSITY_LABELS = {
    "space": "Density (events/kb)",
    "spacetime": "Rate density (events/kb/min)",
}

_FORM_ALIASES: dict[str, tuple[DataForm, str]] = {
    "scalar": ("scalar", "constant"),
    "constant": ("scalar", "constant"),
    "x": ("x", "space"),
    "space": ("x", "space"),
    "t": ("t", "time"),
    "time": ("t", "time"),
    "xt": ("xt", "spacetime"),
    "spacetime": ("xt", "spacetime"),
}

_FORM_LABELS = {
    "scalar": "constant",
    "x": "space",
    "t": "time",
    "xt": "space-time",
    "supplied": "prepared",
    "bundle": "bundle",
    "regions": "regions",
}


def _unique(values: Iterable[T]) -> tuple[T, ...]:
    out: list[T] = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


def _friendly_key(target: str) -> str:
    return target[:-3] if target.endswith("_xt") else target


def _friendly_label(name: str) -> str:
    return _FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())


def _target_records(table: Any) -> tuple[dict[str, Any], ...]:
    if hasattr(table, "to_dict"):
        return tuple(table.to_dict(orient="records"))
    return tuple(dict(record) for record in table)


def _normalise_kind(kind: Any) -> DataForm:
    value = str(kind).strip().lower()
    if value not in DATA_FORMS:
        return "supplied"
    return cast(DataForm, value)


def _target_output(spec: Any) -> tuple[DataForm, str]:
    for name, values in spec.assumption_values:
        if name == "target_form" and values:
            return _FORM_ALIASES.get(str(values[0]), ("supplied", str(values[0])))
    if spec.target in {"initiation_rate", "fork_speed"}:
        default_form = dict(spec.default_assumptions).get("target_form", "space")
        return _FORM_ALIASES.get(str(default_form), ("supplied", str(default_form)))
    if spec.target == "origin_efficiency":
        if "simulation" in spec.method:
            return "x", "space"
        return "regions", "regions"
    return _FIXED_OUTPUTS.get(spec.target, ("supplied", "space"))


def _target_shapes(spec: Any) -> tuple[str, ...]:
    if (
        spec.target == "initiation_rate"
        and "target_shape" in (*spec.required_assumptions, *spec.optional_assumptions)
        and dict(spec.default_assumptions).get("target_form") in {"time", "spacetime"}
    ):
        return ("weibull",)
    for name, values in spec.assumption_values:
        if name == "target_shape":
            return tuple(str(value) for value in values)
    default = dict(spec.default_assumptions).get("target_shape")
    return (str(default),) if default is not None else ()


def _input_forms(spec: Any, name: str) -> tuple[DataForm, ...]:
    for pairs in (spec.source_kinds, spec.assumption_kinds):
        for quantity, kinds in pairs:
            if quantity == name:
                return _unique(_normalise_kind(kind) for kind in kinds)
    if name == "fork_densities_xt":
        return ("bundle",)
    if name == "regions":
        return ("regions",)
    return ("supplied",)


def _input_hint(name: str, forms: tuple[DataForm, ...]) -> str:
    if name == "fork_densities_xt":
        return "Select the paired fork-density bundle from a prepared DNAscape result."
    if name == "regions":
        return "Enter (start, end) coordinates directly or upload a two-column region table."
    if forms == ("scalar",):
        return "Enter a constant value directly."
    form_labels = {
        "scalar": "constant",
        "x": "spatial profile",
        "t": "temporal row profile",
        "xt": "position-by-time matrix",
        "supplied": "prepared data",
    }
    accepted = ", ".join(form_labels[form] for form in forms if form in form_labels)
    track_hint = (
        " Space-time data may be one numeric matrix or multiple genomic tracks "
        "ordered by time point."
        if "xt" in forms
        else ""
    )
    return (
        f"Accepted shape{'' if len(forms) == 1 else 's'}: {accepted}. "
        f"{RDATA_UPLOAD_HINT}{track_hint}"
    )


def _formats_for_form(form: DataForm) -> tuple[str, ...]:
    if form in {"bundle", "scalar"}:
        return ()
    if form in {"x", "xt"}:
        return RDATA_SOURCE_FORMAT_KEYS
    if form == "regions":
        return RDATA_PLAIN_FORMAT_KEYS
    return RDATA_PLAIN_FORMAT_KEYS


def _accept_for_form(form: DataForm) -> str:
    if form in {"bundle", "scalar"}:
        return ""
    if form in {"x", "xt"}:
        return f"{RDATA_PLAIN_UPLOAD_ACCEPT},{RDATA_TRACK_UPLOAD_ACCEPT}"
    if form == "regions":
        return RDATA_PLAIN_UPLOAD_ACCEPT
    return RDATA_PLAIN_UPLOAD_ACCEPT


def _mapping_input(
    spec: Any,
    name: str,
    role: InputRole,
    descriptions: dict[str, str],
    *,
    forms: tuple[DataForm, ...] | None = None,
) -> MappingInput:
    forms = _input_forms(spec, name) if forms is None else forms
    prepared = name == "fork_densities_xt"
    defaults = dict(spec.default_assumptions)
    form_formats: tuple[tuple[DataForm, tuple[str, ...]], ...] = tuple(
        (form, _formats_for_form(form)) for form in forms
    )
    form_accepts: tuple[tuple[DataForm, str], ...] = tuple(
        (form, _accept_for_form(form)) for form in forms
    )
    form_multiples: tuple[tuple[DataForm, bool], ...] = tuple(
        (form, form == "xt") for form in forms
    )
    form_track_modes: tuple[tuple[DataForm, TrackUploadMode], ...] = tuple(
        (form, "single" if form == "x" else "series" if form == "xt" else "unsupported")
        for form in forms
    )
    accepted_formats = _unique(key for _, format_keys in form_formats for key in format_keys)
    accept = ",".join(
        _unique(
            extension
            for _, form_accept in form_accepts
            for extension in form_accept.split(",")
            if extension
        )
    )
    return MappingInput(
        key=name,
        label=_friendly_label(name),
        role=role,
        forms=forms,
        required=role in {"source", "regions"}
        or (name in spec.required_assumptions and name not in defaults),
        default=defaults.get(name),
        default_provided=name in defaults,
        description=_FEATURE_DESCRIPTIONS.get(
            name,
            descriptions.get(name, "DNAscape mapping input."),
        ),
        accepted_formats=accepted_formats,
        accept=accept,
        form_formats=form_formats,
        form_accepts=form_accepts,
        form_multiples=form_multiples,
        form_track_modes=form_track_modes,
        hint=_input_hint(name, forms),
        prepared_result_only=prepared,
    )


def _route_inputs(spec: Any, descriptions: dict[str, str]) -> tuple[MappingInput, ...]:
    inputs = [_mapping_input(spec, name, "source", descriptions) for name in spec.source]
    present = set(spec.source)
    assumption_names = _unique(
        list(spec.required_assumptions) + [name for name, _ in spec.assumption_kinds]
    )
    known_features = (
        set(descriptions)
        | {name for route in (spec,) for name in route.source}
        | set(_FEATURE_LABELS)
    )
    for name in assumption_names:
        if name in present:
            continue
        if name == "regions":
            inputs.append(_mapping_input(spec, name, "regions", descriptions))
        elif name in known_features:
            inputs.append(_mapping_input(spec, name, "feature_assumption", descriptions))
    return tuple(inputs)


def _method_label(inputs: tuple[MappingInput, ...], spec: Any, output_dependence: str) -> str:
    input_labels = []
    for item in inputs:
        role = " setting" if item.role == "feature_assumption" else ""
        forms = "/".join(_FORM_LABELS[form] for form in item.forms)
        input_labels.append(f"{item.label}{role} ({forms})")
    sources = " + ".join(input_labels)
    qualifiers = []
    method = spec.method
    for token, label in (
        ("simulation", "Simulation"),
        ("deterministic", "Deterministic"),
        ("lightcone", "Light-cone"),
        ("mean_field", "Mean field"),
        ("region_integral", "Region integral"),
        ("time_integrated", "Time integrated"),
        ("normalized", "Normalized"),
    ):
        if token in method and label not in qualifiers:
            qualifiers.append(label)
    dependence = output_dependence.capitalize()
    suffix = " | ".join((dependence, *qualifiers))
    return f"{sources} - {suffix}"


def _canonical_signature_name(name: str) -> str:
    return _SIGNATURE_ALIASES.get(name, name)


def _signature_callables(func: Any) -> tuple[Any, ...]:
    """Return a route wrapper and the DNAscape callables it directly delegates to."""

    candidates = [func]
    code = getattr(func, "__code__", None)
    namespace = getattr(func, "__globals__", {})
    if code is not None:
        modules = tuple(
            value
            for name in code.co_names
            if inspect.ismodule(value := namespace.get(name))
            and str(getattr(value, "__name__", "")).startswith("dnascape.")
        )
        for name in code.co_names:
            candidate = namespace.get(name)
            if candidate is None:
                candidate = next(
                    (getattr(module, name) for module in modules if hasattr(module, name)),
                    None,
                )
            if not callable(candidate) or candidate in candidates:
                continue
            module = str(getattr(candidate, "__module__", ""))
            if module == "dnascape" or module.startswith("dnascape."):
                candidates.append(candidate)
    return tuple(candidates)


def _signature_defaults(spec: Any) -> dict[str, Any]:
    """Infer route defaults from the selected wrapper and its direct delegate.

    Most public registry functions are intentionally thin ``**kwargs``
    adapters.  Inspecting the DNAscape callable referenced by that adapter is
    what preserves route-specific defaults such as fit steps and iteration
    counts instead of applying a single generic value to every route.
    """

    # Multiplexing wrappers name both delegates in their bytecode. Inspect
    # only the branch selected by this exact target; temporal and space-time
    # optimizers intentionally use different defaults and inheritance rules.
    target_form = _target_output(spec)[1]
    branch_names = {
        "map_timing_and_speed_to_general_firing": {
            "time": "map_timing_firing_time",
            "spacetime": "map_timing_firing_spacetime",
        },
        "map_timing_and_firing_to_general_speed": {
            "time": "map_timing_firing_temporal_speed",
            "spacetime": "map_timing_firing_spacetime_speed",
        },
    }
    callables = _signature_callables(spec.func)
    selected_branch = branch_names.get(getattr(spec.func, "__name__", ""), {}).get(target_form)
    if selected_branch is not None:
        delegate = next(func for func in callables if func.__name__ == selected_branch)
        callables = (spec.func, *_signature_callables(delegate))
    inferred: dict[str, Any] = {}
    pending = list(callables)
    for func in pending:
        try:
            parameters = tuple(inspect.signature(func).parameters.values())
        except (TypeError, ValueError):
            continue
        if (
            func is not spec.func
            and any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
            and all(parameter.default is inspect.Parameter.empty for parameter in parameters)
        ):
            # Some target projections add another transparent wrapper, e.g.
            # replicated_fraction -> replicated_fraction_xt -> numerical map.
            pending.extend(
                candidate for candidate in _signature_callables(func) if candidate not in pending
            )
        ranked: dict[str, tuple[tuple[int, int], Any]] = {}
        for parameter in parameters:
            if parameter.default is inspect.Parameter.empty:
                continue
            name = _canonical_signature_name(parameter.name)
            score = (
                int(parameter.default is not None),
                int(name == parameter.name),
            )
            current = ranked.get(name)
            if current is None or score > current[0]:
                ranked[name] = (score, parameter.default)
        for name, (_score, value) in ranked.items():
            # An outer None is a meaningful automatic/inherited value. A
            # helper's unrelated numeric default must never override it.
            inferred.setdefault(name, value)
    if selected_branch is not None:
        # Global controls are optional aliases applied before delegation.
        # Sending a fabricated default would override phase-specific settings.
        inferred.update({name: None for name in ("fit_step", "maxiter", "sim_number")})
    for name in set(spec.required_assumptions) | set(spec.optional_assumptions):
        if name not in inferred and name in _SETTING_FALLBACK_DEFAULTS:
            inferred[name] = _SETTING_FALLBACK_DEFAULTS[name]
    # Explicit MappingSpec metadata is authoritative when it intentionally
    # differs from the delegated function signature.
    inferred.update(dict(spec.default_assumptions))
    return inferred


def _setting_label(name: str) -> str:
    if name in _SETTING_LABELS:
        return _SETTING_LABELS[name]
    return name.replace("_", " ").capitalize()


def _setting_kind(name: str, allowed_values: tuple[Any, ...]) -> SettingKind:
    if allowed_values:
        return "choice"
    if name in _BOOLEAN_SETTINGS:
        return "boolean"
    if name in _INTEGER_SETTINGS:
        return "integer"
    if name in _PAIR_SETTINGS:
        return "number_pair"
    if name in _LIST_SETTINGS:
        return "number_list"
    if name in _TEXT_SETTINGS:
        return "text"
    return "number"


def _setting_limits(
    name: str,
    kind: SettingKind,
) -> tuple[float | int | None, float | int | None, float | int | None]:
    if kind == "integer":
        if name == "ffiring_nr":
            return 2, None, 1
        minimum = 0 if name in {"seed", "refine_seed"} else 1
        # Every stochastic fit/forward-route ensemble is exposed consistently.
        # The scientific package itself remains uncapped; 50,000 is the local
        # interactive application's upper bound.
        maximum = 50_000 if "sim_number" in name else None
        return minimum, maximum, 1
    if name == "cfl":
        return 1e-6, 1.0, 0.05
    if name == "stall_rate_per_kb":
        return 0.0, None, 0.001
    if name == "err_threshold":
        return None, None, 0.1
    if name == "initial_initiation_rate":
        return 1e-300, None, 0.001
    if name in _POSITIVE_SETTINGS:
        return 1e-12, None, 0.1
    if name in _NONNEGATIVE_SETTINGS:
        return 0.0, None, 0.1
    return None, None, 0.1 if kind == "number" else None


def _route_settings(spec: Any, inputs: tuple[MappingInput, ...]) -> tuple[MappingSetting, ...]:
    input_names = {item.key for item in inputs}
    assumption_names = _unique(list(spec.required_assumptions) + list(spec.optional_assumptions))
    defaults = _signature_defaults(spec)
    allowed_by_name = {str(name): tuple(values) for name, values in spec.assumption_values}
    has_geometry = "geometry" in assumption_names
    settings: list[MappingSetting] = []
    for name in assumption_names:
        name = str(name)
        if name in input_names or name in _HIDDEN_ASSUMPTIONS:
            continue
        allowed_values = allowed_by_name.get(name, _SETTING_CHOICES.get(name, ()))
        kind = _setting_kind(name, allowed_values)
        minimum, maximum, step = _setting_limits(name, kind)
        has_default = name in defaults
        default = defaults.get(name)
        # The package uses None to distinguish the new physical rate from the
        # legacy per-bin probability. The app exposes only physical units, so
        # zero is the clear, always-visible no-stalling default.
        if name == "stall_rate_per_kb" and default is None:
            default = 0.0
            has_default = True
        # rsim's low-level signature contains its fully expanded 0..1500
        # fallback grid.  The application derives a compact grid from Region
        # controls, so serialising all 1,501 values into each client is both
        # misleading and needlessly expensive.  ``None`` means automatic here.
        if name == "time_grid" and isinstance(default, (tuple, list)) and len(default) > 50:
            default = None
        if name == "tau" and isinstance(default, (int, float)) and default == float("inf"):
            default = None
        if name == "geometry" and default is None:
            default = "periodic" if defaults.get("perQ") else "linear"
            has_default = True
        location = _SETTING_LOCATIONS.get(name, "settings")
        # Mapping uses one Periodic switch for deterministic and stochastic
        # routes. Retain the string alias only for saved/programmatic requests.
        if name == "geometry" and has_geometry:
            location = "derived"
        settings.append(
            MappingSetting(
                key=name,
                label=_setting_label(name),
                category=_SETTING_CATEGORIES.get(name, "fitting"),
                location=location,
                kind=kind,
                required=name in spec.required_assumptions and not has_default,
                has_default=has_default,
                default=default,
                allowed_values=allowed_values,
                description=_SETTING_DESCRIPTIONS.get(
                    name,
                    "Optional setting accepted by this exact DNAscape mapping route.",
                ),
                unit=_SETTING_UNITS.get(name),
                min_value=minimum,
                max_value=maximum,
                step=step,
                nullable=not (name in spec.required_assumptions and not has_default),
            )
        )
    return tuple(settings)


def _build_routes(specs: tuple[Any, ...], descriptions: dict[str, str]) -> tuple[MappingRoute, ...]:
    routes: list[MappingRoute] = []
    for spec in specs:
        output_form, output_dependence = _target_output(spec)
        inputs = _route_inputs(spec, descriptions)
        if len(inputs) == 1 and _friendly_key(inputs[0].key) == _friendly_key(spec.target):
            # Mapping must perform a conversion. Keep reductions such as
            # f(x,t) -> f(t), but never offer the already supplied target in
            # the same form. Rebuild all form-dependent upload descriptors
            # if a future registry method accepts both kinds of input.
            source = inputs[0]
            forms: tuple[DataForm, ...] = tuple(
                form for form in source.forms if form != output_form
            )
            if not forms:
                continue
            if forms != source.forms:
                inputs = (_mapping_input(spec, source.key, source.role, descriptions, forms=forms),)
        settings = _route_settings(spec, inputs)
        shapes = _target_shapes(spec)
        defaults = tuple((str(name), value) for name, value in spec.default_assumptions)
        assumption_forms: tuple[tuple[str, tuple[DataForm, ...]], ...] = tuple(
            (str(name), tuple(_normalise_kind(kind) for kind in kinds))
            for name, kinds in spec.assumption_kinds
        )
        assumption_values = tuple(
            (str(name), tuple(values)) for name, values in spec.assumption_values
        )
        routes.append(
            MappingRoute(
                target=spec.target,
                target_key=_friendly_key(spec.target),
                method=spec.method,
                label=_method_label(inputs, spec, output_dependence),
                description=spec.description or descriptions.get(spec.target, ""),
                inputs=inputs,
                settings=settings,
                output_form=output_form,
                output_dependence=output_dependence,
                target_shapes=shapes,
                default_target_shape=dict(defaults).get("target_shape"),
                required_assumptions=tuple(str(name) for name in spec.required_assumptions),
                optional_assumptions=tuple(str(name) for name in spec.optional_assumptions),
                defaults=defaults,
                assumption_forms=assumption_forms,
                assumption_values=assumption_values,
                prepared_result_only=spec.target == "fork_densities_xt",
            )
        )
    return tuple(routes)


def _merge_family_input(
    key: str,
    variants: tuple[MappingRoute, ...],
    *,
    required: bool,
) -> MappingInput:
    """Merge exact descriptors for one factor without weakening route constraints."""

    descriptors = tuple(
        item for route in variants for item in route.inputs if _friendly_key(item.key) == key
    )
    prototype = descriptors[0]
    forms: tuple[DataForm, ...] = tuple(
        form for form in DATA_FORMS if any(form in descriptor.forms for descriptor in descriptors)
    )
    form_formats: tuple[tuple[DataForm, tuple[str, ...]], ...] = tuple(
        (form, _formats_for_form(form)) for form in forms
    )
    form_accepts: tuple[tuple[DataForm, str], ...] = tuple(
        (form, _accept_for_form(form)) for form in forms
    )
    form_multiples: tuple[tuple[DataForm, bool], ...] = tuple(
        (form, form == "xt") for form in forms
    )
    form_track_modes: tuple[tuple[DataForm, TrackUploadMode], ...] = tuple(
        (form, "single" if form == "x" else "series" if form == "xt" else "unsupported")
        for form in forms
    )
    defaults = tuple(item.default for item in descriptors if item.default_provided)
    accepted_formats = _unique(format_key for _, keys in form_formats for format_key in keys)
    accept = ",".join(
        _unique(
            extension
            for _, form_accept in form_accepts
            for extension in form_accept.split(",")
            if extension
        )
    )
    role: InputRole
    if any(item.role == "regions" for item in descriptors):
        role = "regions"
    elif any(item.role == "source" for item in descriptors):
        role = "source"
    else:
        role = "feature_assumption"
    return MappingInput(
        key=key,
        label=(
            f"{_friendly_label(key)} ({'/'.join(_FORM_LABELS[form] for form in forms)})"
            if key == variants[0].target_key
            else _friendly_label(key)
        ),
        role=role,
        forms=forms,
        required=required,
        default=defaults[0] if defaults else None,
        default_provided=bool(defaults),
        description=prototype.description,
        accepted_formats=accepted_formats,
        accept=accept,
        form_formats=form_formats,
        form_accepts=form_accepts,
        form_multiples=form_multiples,
        form_track_modes=form_track_modes,
        hint=_input_hint(key, forms),
        prepared_result_only=all(item.prepared_result_only for item in descriptors),
    )


def _build_route_families(routes: tuple[MappingRoute, ...]) -> tuple[MappingRouteFamily, ...]:
    """Group exact methods by target/dependence and biological factor names."""

    grouped: dict[tuple[str, str, tuple[str, ...]], list[MappingRoute]] = {}
    for route in routes:
        # A package default changes how a factor is supplied, not which
        # biological factors the mapping needs. Keep convenience wrappers in
        # the same family as their explicit-source counterparts.
        factor_keys = tuple(_friendly_key(item.key) for item in route.inputs)
        signature = (
            route.target_key,
            route.output_dependence,
            tuple(sorted(factor_keys)),
        )
        grouped.setdefault(signature, []).append(route)

    families: list[MappingRouteFamily] = []
    for (
        target_key,
        dependence,
        _factor_signature,
    ), grouped_variants in grouped.items():
        variants = tuple(grouped_variants)
        internal_targets = _unique(route.target for route in variants)
        output_forms: tuple[DataForm, ...] = _unique(route.output_form for route in variants)
        if len(internal_targets) != 1 or len(output_forms) != 1:
            raise RuntimeError(
                "A mapping route family crossed an internal target or output form boundary."
            )
        input_keys = _unique(_friendly_key(item.key) for route in variants for item in route.inputs)
        required_keys = input_keys
        inputs = tuple(
            _merge_family_input(key, variants, required=key in required_keys) for key in input_keys
        )
        factor_label = " + ".join(
            next(item.label for item in inputs if item.key == key) for key in required_keys
        )
        target_shapes = _unique(shape for route in variants for shape in route.target_shapes)
        defaults = _unique(
            route.default_target_shape
            for route in variants
            if route.default_target_shape is not None
        )
        key_suffix = "-and-".join(required_keys) or "no-inputs"
        families.append(
            MappingRouteFamily(
                key=f"{target_key}--{dependence}--{key_suffix}",
                target=internal_targets[0],
                target_key=target_key,
                label=factor_label or "No data inputs",
                description=f"Map from {factor_label.lower()}.",
                inputs=inputs,
                required_factor_keys=required_keys,
                variants=variants,
                output_form=output_forms[0],
                output_dependence=dependence,
                target_shapes=tuple(str(shape) for shape in target_shapes),
                default_target_shape=str(defaults[0]) if defaults else None,
                prepared_result_only=all(route.prepared_result_only for route in variants),
            )
        )
    return tuple(families)


def _variant_description(
    internal_target: str,
    dependence: str,
    descriptions: dict[str, str],
) -> str:
    if internal_target in {"initiation_events", "initiation_events_xt", "coalescence_events", "coalescence_events_xt"}:
        return _FEATURE_DESCRIPTIONS[internal_target]
    description = descriptions.get(internal_target, "Registered DNAscape mapping target.")
    return f"{description} Output dependence: {dependence}."


def _build_targets(
    public_target_ids: tuple[str, ...],
    descriptions: dict[str, str],
    routes: tuple[MappingRoute, ...],
) -> tuple[MappingTarget, ...]:
    group_order = _unique(_friendly_key(target) for target in public_target_ids)
    targets: list[MappingTarget] = []
    for key in group_order:
        grouped_routes = tuple(route for route in routes if route.target_key == key)
        dependences = tuple(
            sorted(
                _unique(route.output_dependence for route in grouped_routes),
                key=lambda item: (
                    DEPENDENCE_ORDER.index(item)
                    if item in DEPENDENCE_ORDER
                    else len(DEPENDENCE_ORDER),
                    item,
                ),
            )
        )
        variants: list[TargetVariant] = []
        for dependence in dependences:
            variant_routes = tuple(
                route for route in grouped_routes if route.output_dependence == dependence
            )
            internal_targets = _unique(route.target for route in variant_routes)
            internal_target = internal_targets[0]
            output_forms: tuple[DataForm, ...] = _unique(
                route.output_form for route in variant_routes
            )
            shapes = _unique(shape for route in variant_routes for shape in route.target_shapes)
            defaults = _unique(
                route.default_target_shape
                for route in variant_routes
                if route.default_target_shape is not None
            )
            variants.append(
                TargetVariant(
                    internal_target=internal_target,
                    dependence=dependence,
                    label=(_EVENT_DENSITY_LABELS[dependence]
                           if key in {"initiation_events", "coalescence_events"}
                           else _DEPENDENCE_LABELS.get(dependence, dependence.capitalize())),
                    output_form=output_forms[0],
                    target_shapes=tuple(str(shape) for shape in shapes),
                    default_target_shape=str(defaults[0]) if defaults else None,
                    description=_variant_description(internal_target, dependence, descriptions),
                    methods=tuple(route.method for route in variant_routes),
                    prepared_result_only=internal_target == "fork_densities_xt",
                )
            )

        preferred_description_target = key if key in descriptions else variants[0].internal_target
        default_dependence = _DEFAULT_DEPENDENCE.get(key, variants[0].dependence)
        if default_dependence not in dependences:
            default_dependence = variants[0].dependence
        targets.append(
            MappingTarget(
                key=key,
                label=_friendly_label(key),
                description=_TARGET_GROUP_DESCRIPTIONS.get(
                    key,
                    descriptions.get(
                        preferred_description_target,
                        variants[0].description,
                    ),
                ),
                variants=tuple(variants),
                default_dependence=default_dependence,
                prepared_result_only=all(item.prepared_result_only for item in variants),
                requires_simulation=all(route.method.endswith("_simulation") for route in grouped_routes),
            )
        )
    return tuple(targets)


@lru_cache(maxsize=1)
def get_mapping_catalog() -> MappingCatalog:
    """Build and cache the deterministic Mapping application catalogue.

    Both scientific discovery calls are public.  The returned snapshot is
    independent of rmap's mutable internal registration list.
    """

    from dnascape import rmap

    target_table = rmap.targets(implemented=True)
    target_records = _target_records(target_table)
    public_target_ids = tuple(str(record["target"]) for record in target_records)
    descriptions = {
        str(record["target"]): str(record.get("description", "")) for record in target_records
    }
    public_targets = set(public_target_ids)
    specs = tuple(
        spec for spec in rmap.routes() if spec.implemented and spec.target in public_targets
    )
    # Simulation retains its own observables and APIs. Mapping presents only
    # deterministic conversions, without probability-density or identity routes.
    removed_targets = {"initiation_pdf", "initiation_pdf_xt", "coalescence_pdf", "coalescence_pdf_xt"}
    removed_methods = {
        "fork_density_bundle_to_directionality_time_integrated",
        "fork_density_fields_to_directionality_time_integrated",
        "fork_density_profiles_to_directionality",
        "replicated_fraction_xt_to_replicated_fraction_spatial_mean",
    }
    specs = tuple(spec for spec in specs
                  if spec.target not in removed_targets
                  and spec.method not in removed_methods
                  and not spec.method.endswith("_simulation"))
    public_target_ids = tuple(target for target in public_target_ids
                              if any(spec.target == target for spec in specs))
    methods = tuple(spec.method for spec in specs)
    if len(methods) != len(set(methods)):
        raise RuntimeError("rmap returned duplicate implemented method names.")
    missing = tuple(
        target for target in public_target_ids if not any(spec.target == target for spec in specs)
    )
    if missing:
        raise RuntimeError(f"Public rmap targets have no implemented routes: {', '.join(missing)}")

    routes = _build_routes(specs, descriptions)
    # A target with only a pass-through method is not a Mapping operation.
    public_target_ids = tuple(
        target for target in public_target_ids if any(route.target == target for route in routes)
    )
    route_families = _build_route_families(routes)
    targets = _build_targets(public_target_ids, descriptions, routes)
    return MappingCatalog(
        targets=targets,
        routes=routes,
        route_families=route_families,
        public_target_ids=public_target_ids,
    )


def mapping_targets() -> tuple[MappingTarget, ...]:
    """Return the friendly, dependence-aware target choices."""

    return get_mapping_catalog().targets


def mapping_target(key: str) -> MappingTarget:
    """Return a target by friendly key or exact internal target."""

    return get_mapping_catalog().target(key)


def mapping_routes(
    target: str | None = None,
    dependence: str | None = None,
    *,
    internal_target: str | None = None,
) -> tuple[MappingRoute, ...]:
    """Return deterministic route choices for the requested target filters."""

    return get_mapping_catalog().routes_for(
        target=target,
        dependence=dependence,
        internal_target=internal_target,
    )


def mapping_route(method: str) -> MappingRoute:
    """Return an exact route by its public rmap method name."""

    return get_mapping_catalog().route(method)


def mapping_route_families(
    target: str | None = None,
    dependence: str | None = None,
    *,
    internal_target: str | None = None,
) -> tuple[MappingRouteFamily, ...]:
    """Return user-facing factor-combination route families."""

    return get_mapping_catalog().route_families_for(
        target=target,
        dependence=dependence,
        internal_target=internal_target,
    )


def mapping_route_family(key: str) -> MappingRouteFamily:
    """Return one route family by its stable key."""

    return get_mapping_catalog().route_family(key)


def rdata_formats(*, source: bool = True, target: bool = False) -> tuple[RDataFormat, ...]:
    """Return rdata format metadata for upload and/or download controls."""

    return tuple(
        item
        for item in RDATA_FORMATS
        if (source and item.source_supported) or (target and item.target_supported)
    )


__all__ = [
    "DATA_FORMS",
    "DEPENDENCE_ORDER",
    "RDATA_FORMATS",
    "RDATA_FORM_HINT",
    "RDATA_PLAIN_FORMAT_KEYS",
    "RDATA_PLAIN_UPLOAD_ACCEPT",
    "RDATA_SOURCE_FORMAT_KEYS",
    "RDATA_TARGET_FORMAT_KEYS",
    "RDATA_TRACK_FORMAT_KEYS",
    "RDATA_TRACK_UPLOAD_ACCEPT",
    "RDATA_UPLOAD_ACCEPT",
    "RDATA_UPLOAD_HINT",
    "MappingCatalog",
    "MappingInput",
    "MappingRoute",
    "MappingRouteFamily",
    "MappingSetting",
    "MappingTarget",
    "RDataFormat",
    "SettingCategory",
    "SettingKind",
    "SettingLocation",
    "TargetVariant",
    "TrackUploadMode",
    "get_mapping_catalog",
    "mapping_route",
    "mapping_route_families",
    "mapping_route_family",
    "mapping_routes",
    "mapping_target",
    "mapping_targets",
    "rdata_formats",
]
