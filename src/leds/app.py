from __future__ import annotations

import importlib.resources
import threading
import traceback
from contextlib import contextmanager
from functools import partial, wraps

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import FixedTicker

from leds import (
    dataset_view,
    event_details,
    sipm_view,
    spectrum_view,
    validation,
    validation_view,
)
from leds.all_waveforms_view import (
    SYSTEMS,
    AllWaveformsFigure,
    category_options,
    groupings_for,
    raw_kinds_for,
    y_axis_label,
)
from leds.array_view import build_source_data, empty_source, make_event_figure
from leds.config import discover_cycles, resolve_base_paths
from leds.event_viewer import EventViewer, parse_timestamp
from leds.spectrum import (
    BINARY_CUTS,
    DEFAULT_BIN_WIDTH,
    ENERGY_RANGE,
    MULT_OPTIONS,
    N_BINS,
    RunSpectrum,
    bins_for_width,
)
from leds.waveform_proc import PROCESSED_PARAMS, WaveformProcessor
from leds.waveform_view import RAW, plot_event_waveforms

pn.extension("tabulator", sizing_mode="stretch_width")

# Match the LEGEND monitoring dashboard's branding.
LEGEND_LOGO = "https://legend-exp.org/typo3conf/ext/sitepackage/Resources/Public/Images/Logo/logo_legend_tag_next.svg"
LEGEND_FAVICON = "https://legend-exp.org/typo3conf/ext/sitepackage/Resources/Public/Favicons/android-chrome-96x96.png"
HEADER_BACKGROUND = "#f8f8fa"
HEADER_COLOR = "#1A2A5B"

# tab order (used for both layout and the lazy per-tab update gating)
TAB_EVENT, TAB_DETAILS, TAB_WAVEFORMS, TAB_SPECTRUM, TAB_DATASET, TAB_VALIDATION = (
    0,
    1,
    2,
    3,
    4,
    5,
)

_LOGO_DIR = importlib.resources.files("leds") / "logos"
_HEADER_LINKS = (
    ("github-mark.png", "https://github.com/legend-exp/", 24),
    ("logo_indico.png", "https://indico.legend-exp.org", 24),
    (
        "confluence.png",
        "https://legend-exp.atlassian.net/wiki/spaces/LEGEND/overview",
        24,
    ),
    ("elog.png", "https://elog.legend-exp.org/ELOG/", 30),
)


def _vertical_label_html(text):
    """A vertically-centred, bottom-to-top rotated label (shared grid y-axis)."""
    return (
        '<div style="height:100%;display:flex;align-items:center;'
        'justify-content:center;">'
        '<div style="writing-mode:vertical-rl;transform:rotate(180deg);'
        f'font-size:11px;white-space:nowrap;color:#444;">{text}</div></div>'
    )


def user_chip(user):
    """Header HTML showing who is signed in, with a logout link.

    ``None`` when the server runs without authentication (no user, or Panel's
    "guest" placeholder) so the header stays unchanged.
    """
    if not user or user == "guest":
        return None
    import html  # noqa: PLC0415 (tiny stdlib helper, login-only path)

    name = html.escape(user)
    # the anchor is styled as a button to match the login page's buttons
    return (
        f'<span style="color:{HEADER_COLOR};font-size:0.9em;white-space:nowrap;'
        'display:inline-flex;align-items:center;gap:10px;">'
        f"{name}"
        f'<a href="./logout" style="background:{HEADER_COLOR};color:#ffffff;'
        "border:1px solid #1a2a5b;border-radius:4px;padding:4px 12px;"
        'text-decoration:none;font-size:0.95em;">Log out</a></span>'
    )


def _update_energy(source, data):
    """Send only the ``energy`` column when the rest of ``data`` is unchanged.

    Falls back to a full assignment whenever the column lengths disagree --
    ``patch`` requires them to match, and a changed detector count means the
    static geometry changed too.
    """
    energies = data["energy"]
    current = source.data.get("energy")
    if current is not None and len(current) == len(energies):
        source.patch({"energy": [(slice(len(energies)), list(energies))]})
    else:
        source.data = data


def _serialized(method):
    """Run an :class:`EventDisplay` entry point under its session lock.

    With ``pn.config.nthreads`` set, Panel runs widget events, Bokeh events,
    periodic callbacks and ``onload`` on a thread pool, so two callbacks of the
    *same* session can otherwise overlap and race on the per-session state
    (the viewer's caches and open files, the accumulating spectrum, the tab
    bookkeeping). One re-entrant lock per session serialises them; nested
    watchers -- a handler setting a param whose own watcher fires -- re-enter
    it on the same thread. Direct Bokeh writes inside go through
    :meth:`EventDisplay._on_loop`. Without ``nthreads`` the lock is never
    contended and every callback behaves exactly as before.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    wrapper._serialized = True
    return wrapper


def build_header_links():
    """Row of LEGEND resource icon-links, as in the monitoring dashboard header."""
    return pn.Row(
        *(
            pn.pane.Image(
                str(_LOGO_DIR / filename),
                link_url=url,
                fixed_aspect=True,
                width=width,
            )
            for filename, url, width in _HEADER_LINKS
        ),
        align="center",
    )


class EventDisplay(param.Parameterized):
    """Interactive detector-array event view backed by an :class:`EventViewer`.

    One instance per user session (built by :func:`create_app`), so the mutable
    per-event state is never shared across the hosted server's clients. The
    array is a Bokeh patches glyph; each event swaps the ColumnDataSource data
    rather than re-rendering an image.
    """

    production_cycle = param.Selector(default=None, objects=[])
    period = param.Selector(default=None, objects=[])
    run = param.Selector(default=None, objects=[])
    index = param.Integer(default=0, bounds=(0, None))
    playing = param.Boolean(default=False)
    playback_period = param.Integer(default=200, bounds=(50, 2000))
    waveform_param = param.Selector(default=RAW, objects=[RAW, *PROCESSED_PARAMS])
    subtract_baseline = param.Boolean(default=True)
    show_waveforms = param.Boolean(default=False)
    show_spectrum = param.Boolean(default=False)
    selected_detector = param.String(default="")
    all_wf_system = param.Selector(default="geds", objects=list(SYSTEMS))
    all_wf_grouping = param.Selector(default="String", objects=groupings_for("geds"))
    all_wf_category = param.Selector(default="all", objects=["all"])
    all_wf_exploded = param.Boolean(default=False)
    all_wf_kind = param.Selector(
        default="waveform_windowed", objects=raw_kinds_for("geds")
    )

    # whole-run Spectrum tab cuts + bin width. Each binary cut is a pair of
    # checkboxes (positive/negative option); multiplicity is off/1/2/>2.
    spectrum_bin_width = param.Number(default=DEFAULT_BIN_WIDTH, bounds=(0.1, 10.0))
    cut_geds_trigger_forced = param.Boolean(default=False)
    cut_geds_trigger_normal = param.Boolean(default=False)
    cut_muon_coincident = param.Boolean(default=False)
    cut_muon_anticoincident = param.Boolean(default=False)
    cut_spms_coincident = param.Boolean(default=False)
    cut_spms_anticoincident = param.Boolean(default=False)
    cut_quality_pass = param.Boolean(default=False)
    cut_quality_fail = param.Boolean(default=False)
    cut_psd_pass = param.Boolean(default=False)
    cut_psd_fail = param.Boolean(default=False)
    cut_multiplicity = param.Selector(default="off", objects=list(MULT_OPTIONS))

    # Dataset tab: which metadata matrix, for which datatype
    dataset_plot = param.Selector(
        default=dataset_view.PLOTS[0], objects=list(dataset_view.PLOTS)
    )
    dataset_datatype = param.Selector(
        default="phy", objects=list(dataset_view.DATATYPES)
    )

    # Validation tab: which check, the rate bin width, and the detector shown
    # in the calibration-detail plot
    validation_plot = param.Selector(
        default=validation_view.PLOTS[0], objects=list(validation_view.PLOTS)
    )
    validation_bin_width = param.Selector(
        default=validation.DEFAULT_BIN_WIDTH, objects=list(validation.BIN_WIDTHS)
    )
    validation_log_y = param.Boolean(default=True)
    validation_string = param.Selector(default="all strings", objects=["all strings"])
    validation_detector = param.Selector(default=None, objects=[])

    def __init__(self, base_path=None, **params):
        # serialises this session's callbacks; see _serialized. Created first:
        # the watchers it wraps are registered by super().__init__, and
        # _relayout() is called below
        self._lock = threading.RLock()
        # the thread that owns this session's document: Bokeh builds the
        # session on its event loop, so anything later running on another
        # thread is one of Panel's pool threads (see _on_loop)
        self._loop_thread = threading.get_ident()
        self.roots = resolve_base_paths(base_path)
        self._cycle_paths = discover_cycles(self.roots)
        if not self._cycle_paths:
            # no dataflow-config found under any root; expose the roots anyway so
            # the load error is shown in-app rather than crashing at startup
            self._cycle_paths = {r.name: r for r in self.roots}
        cycles = list(self._cycle_paths)
        self.param.production_cycle.objects = cycles
        params.setdefault("production_cycle", cycles[0])

        self._cycle_error = None
        self._load_cycle(params["production_cycle"])

        periods = sorted(self.runs)
        self.param.period.objects = periods
        if periods:
            params.setdefault("period", periods[0])

        # guards our own data watchers while params are updated internally
        # (widget syncing still sees the events)
        self._internal_change = False
        # set while one action re-points several all-waveforms params at once
        self._suspend_all_wf = False
        super().__init__(**params)

        self.geds_source = empty_source()
        self.sipm_source = sipm_view.empty_source()
        self.figure, self.glyph = make_event_figure(self.geds_source, self.sipm_source)
        self.bokeh_pane = pn.pane.Bokeh(self.figure, sizing_mode="stretch_both")
        self.geds_source.selected.on_change("indices", self._on_tap)

        self.wf_pane = pn.pane.Bokeh(sizing_mode="stretch_both")
        self.processor = WaveformProcessor(self.viewer)

        self.run_spectrum = RunSpectrum(self.viewer)
        self.validation_data = validation.ValidationData(self.viewer)
        self.spectrum_source = spectrum_view.empty_source()
        self.spectrum_pane = pn.pane.Bokeh(
            spectrum_view.make_figure(self.spectrum_source), sizing_mode="stretch_both"
        )

        self.main_row = pn.Row(
            self.bokeh_pane, sizing_mode="stretch_both", min_height=600
        )

        self.all_wf_pane = pn.pane.Bokeh(sizing_mode="stretch_both", min_height=600)
        self.all_wf_ylabel = pn.pane.HTML(
            "", width=22, sizing_mode="stretch_height", margin=0
        )
        # holds either [pane] (compressed) or [ylabel, pane] (exploded)
        self.all_wf_area = pn.Row(
            self.all_wf_pane, sizing_mode="stretch_both", min_height=600
        )
        self.exploded_toggle = pn.widgets.Toggle.from_param(
            self.param.all_wf_exploded, name="Exploded", width=110
        )
        self.all_wf_tab = pn.Column(
            pn.Row(
                pn.widgets.Select.from_param(
                    self.param.all_wf_system, name="System", width=110
                ),
                pn.widgets.Select.from_param(
                    self.param.all_wf_grouping, name="Group by", width=140
                ),
                pn.widgets.Select.from_param(
                    self.param.all_wf_category, name="Show", width=160
                ),
                pn.widgets.Select.from_param(
                    self.param.all_wf_kind, name="Raw waveform", width=180
                ),
                self.exploded_toggle,
            ),
            self.all_wf_area,
            sizing_mode="stretch_both",
            min_height=600,
        )
        self.run_spectrum_source = spectrum_view.empty_source()
        self.run_spectrum_pane = pn.pane.Bokeh(
            spectrum_view.make_figure(self.run_spectrum_source),
            sizing_mode="stretch_both",
            min_height=600,
        )
        spectrum_tab = pn.Column(
            pn.widgets.FloatSlider.from_param(
                self.param.spectrum_bin_width,
                name="Bin width (keV)",
                step=0.1,
                width=240,
                # only re-histogram on mouse-up: an unthrottled drag re-cuts and
                # re-flattens the whole run once per intermediate step
                throttled=True,
            ),
            pn.Row(
                self._cut_column("geds_trigger"),
                self._cut_column("muon"),
                self._cut_column("quality"),
                pn.Column(
                    pn.pane.Markdown("**multiplicity**", margin=(0, 5)),
                    pn.widgets.Select.from_param(
                        self.param.cut_multiplicity, name="", width=90
                    ),
                ),
                self._cut_column("spms"),
                self._cut_column("psd"),
            ),
            self.run_spectrum_pane,
            sizing_mode="stretch_both",
            min_height=600,
        )

        self.summary_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["field", "value"]),
            disabled=True,
            show_index=False,
            sizing_mode="stretch_width",
        )
        detail_heights = {"coincident": 200, "trigger": 190, "geds": 480, "spms": 320}
        self.detail_tables = {
            name: pn.widgets.Tabulator(
                pd.DataFrame(columns=["field", "value"]),
                disabled=True,
                show_index=False,
                sizing_mode="stretch_width",
                height=detail_heights.get(name, 300),
            )
            for name in event_details.TABLES
        }
        details_tab = pn.Column(
            pn.Column("### summary", self.summary_table),
            *(
                pn.Column(f"### evt/{name}", self.detail_tables[name])
                for name in event_details.TABLES
            ),
            sizing_mode="stretch_width",
            min_height=600,
            scroll=True,
        )

        self.dataset_pane = pn.pane.Bokeh(sizing_mode="stretch_width")
        self._dataset_figs: dict = {}  # (datatype, plot) -> figure, per cycle
        self.dataset_datatype_select = pn.widgets.Select.from_param(
            self.param.dataset_datatype, name="Datatype", width=90
        )
        self.dataset_tab = pn.Column(
            pn.Row(
                pn.widgets.Select.from_param(
                    self.param.dataset_plot, name="Show", width=140
                ),
                self.dataset_datatype_select,
            ),
            self.dataset_pane,
            sizing_mode="stretch_width",
            min_height=600,
            scroll=True,
        )

        self.validation_pane = pn.pane.Bokeh(sizing_mode="stretch_width")
        self._validation_figs: dict = {}  # per-period/plot figures, per cycle
        self._cal_cache: dict = {}  # (cycle, period, run) -> calibration residuals
        self.validation_bin_select = pn.widgets.Select.from_param(
            self.param.validation_bin_width, name="Bin width", width=90
        )
        self.validation_log_toggle = pn.widgets.Checkbox.from_param(
            self.param.validation_log_y, name="Log y", margin=(28, 5, 5, 5)
        )
        self.validation_string_select = pn.widgets.Select.from_param(
            self.param.validation_string, name="String", width=100
        )
        self.validation_detector_select = pn.widgets.Select.from_param(
            self.param.validation_detector, name="Detector", width=120, visible=False
        )
        self.validation_tab = pn.Column(
            pn.Row(
                pn.widgets.Select.from_param(
                    self.param.validation_plot, name="Show", width=170
                ),
                self.validation_bin_select,
                self.validation_string_select,
                self.validation_log_toggle,
                self.validation_detector_select,
            ),
            self.validation_pane,
            sizing_mode="stretch_width",
            min_height=600,
            scroll=True,
        )

        self.tabs = pn.Tabs(
            ("Event display", self.main_row),
            ("Event details", details_tab),
            ("All waveforms", self.all_wf_tab),
            ("Spectrum", spectrum_tab),
            ("Dataset", self.dataset_tab),
            ("Validation", self.validation_tab),
            # static, not dynamic: every tab stays mounted and switching is
            # purely client-side. Dynamic tabs re-render the child on the
            # server for every switch, and rapid back-and-forth on a slow
            # deployment left the returning tab's plot blank. The tabs'
            # content is still built lazily -- each updater gates on the
            # active tab -- and a build that finishes after the user has moved
            # on lands in the (mounted) hidden tab, ready for their return.
            dynamic=False,
        )

        self.tabs.param.watch(self._on_tab, "active")

        self.message = pn.pane.Alert("", alert_type="warning", visible=False)

        self.prev_button = pn.widgets.Button(name="◀ Previous", width=110)
        self.next_button = pn.widgets.Button(name="Next ▶", width=110)
        self.prev_button.on_click(self._on_prev)
        self.next_button.on_click(self._on_next)
        self.timestamp_input = pn.widgets.TextInput(
            name="Jump to timestamp", placeholder="unix"
        )
        self.find_button = pn.widgets.Button(name="Find event", width=110)
        self.find_button.on_click(self._on_find)
        self.play_toggle = pn.widgets.Toggle.from_param(
            self.param.playing, name="▶ Play run", width=110
        )

        # spectrum-bin event selection: when set, next/prev iterate these indices
        self.event_selection = None
        self.selection_info = pn.pane.Markdown("")
        self.clear_button = pn.widgets.Button(name="Clear selection", width=140)
        self.clear_button.on_click(self._on_clear)
        self.run_spectrum_source.selected.on_change("indices", self._on_bin_select)

        self._playback_cb = None
        self._run_length = None

        # last state each tab was drawn for; see _tab_is_current
        self._tab_state: dict = {}

        # channelmap the event sources' static columns were built for
        self._source_tstamp = None

        # running accumulating-spectrum counts; see _accumulated_histogram
        self._accum_scope = None
        self._accum_index = None
        self._accum_counts = None

        # the all-waveforms figure, rebuilt only when its layout changes
        self._all_wf_figure = AllWaveformsFigure()

        self._apply_period(defer=True)
        self._relayout()
        # hand the page over first, then read event 0
        # (the spinner goes on the row, not the Bokeh pane: setting `loading`
        # on a Bokeh pane inside a template raises in Panel 1.9's
        # _sync_properties, which would abort the rest of __init__)
        self.main_row.loading = True
        pn.state.onload(self._first_render)

    def _on_loop(self, write):
        """Run ``write`` -- direct Bokeh model writes -- under the document lock.

        Bokeh models may only be changed while their document is locked. On
        the event loop a callback already holds it, so ``write`` runs at once;
        on one of Panel's pool threads it is queued as a next-tick callback,
        which Bokeh runs on the loop with the lock. Either way the writes run
        inside Panel's ``unlocked()``, so they are combined into one patch and
        dispatched through the same writer Panel uses for its own model
        updates -- mixing Bokeh's flush with Panel's deferred writer let a
        data patch reach the browser before the figure it belonged to. This
        is the pattern Panel itself uses for model updates from a thread.
        (Panel's threaded ``hold()`` is not used: with two callbacks of one
        session on two pool threads, the loop can unhold the document while
        the second is still writing.) The heavy work -- reads, histogramming
        -- stays on the thread; only the final assignments move, so callers
        compute first and pass a closure over the results.
        """
        doc = pn.state.curdoc
        if doc is None or doc.session_context is None:
            write()
            return

        def dispatch():
            with pn.io.unlocked():
                write()

        if threading.get_ident() == self._loop_thread:
            dispatch()
        else:
            doc.add_next_tick_callback(dispatch)

    def _dispatch(self, method, *args):
        """Run a Bokeh-invoked handler where Panel runs everything else.

        Bokeh calls ``on_change`` handlers on the event loop, holding the
        document lock. Under ``nthreads`` that is the one place a slow read
        would still freeze every session in the worker, so hand the work to
        Panel's pool; without one, run it inline as before.
        """
        if pn.config.nthreads:
            pn.state.execute(partial(method, *args), schedule="thread")
        else:
            method(*args)

    @_serialized
    def _on_tab(self, _event):
        # each updater self-gates on the active tab and on whether anything
        # it draws has actually changed, so this is one rebuild at most; each
        # runs on its own so one tab's failure cannot leave another half-drawn
        for label, update in (
            ("all waveforms", self._update_all_waveforms),
            ("spectrum", self._update_run_spectrum),
            ("event details", self._update_event_details),
            ("dataset", self._update_dataset),
            ("validation", self._update_validation),
        ):
            self._guarded(label, update)

    def _guarded(self, label, update):
        """Run one tab updater; an unexpected failure becomes a message.

        The updaters already turn the errors they expect (missing files, bad
        keys) into the Alert. Anything else used to propagate out of the tab
        watcher, aborting it half-way through Panel's own tab switch, which is
        how a tab can end up with its controls but no plot. Show it instead,
        and print the traceback so the container log has it.
        """
        try:
            update()
        except Exception as exc:
            traceback.print_exc()
            self.message.object = f"**{label}:** {type(exc).__name__}: {exc}"
            self.message.visible = True

    @_serialized
    def _on_clear(self, _event):
        self._clear_selection()

    def _tab_is_current(self, tab, key):
        """True when ``tab`` is already drawn for exactly this state.

        Cheaper and far less error-prone than invalidating on every mutation:
        each updater declares what it draws from, and re-drawing is skipped
        while that is unchanged. Switching away and back is then free, and a
        forgotten invalidation point cannot leave a stale pane on screen.

        The state is only recorded once the draw succeeds (``_tab_drawn``), so
        a failed build is retried rather than remembered as done.
        """
        return self._tab_state.get(tab) == key

    def _tab_drawn(self, tab, key):
        self._tab_state[tab] = key

    # -- reactive plumbing ----------------------------------------------------
    #
    # period/run/index changes funnel through exactly one _render each:
    # user-facing params are updated via _set_quietly (which suppresses our
    # own watchers, but not the widget syncing) and the explicit
    # _after_data_change/_render calls do the work once.

    def _load_cycle(self, cycle):
        """Build the viewer for ``cycle``; an incompatible cycle is not fatal."""
        try:
            self.viewer = EventViewer(self._cycle_paths[cycle])
            self.runs = self.viewer.available_runs()
            self._cycle_error = None
        except Exception as exc:
            self.viewer = None
            self.runs = {}
            self._cycle_error = f"{type(exc).__name__}: {exc}"

    def _set_quietly(self, **kwargs):
        """Update params without re-triggering the data watchers below."""
        self._internal_change = True
        try:
            self.param.update(**kwargs)
        finally:
            self._internal_change = False

    def _apply_period(self, *, defer=False):
        """Refresh the run options for the current period and load its event 0.

        With ``defer`` the run options -- cheap, and needed for the sidebar to
        be right in the first paint -- are set inline, while the event read is
        left to :meth:`_first_render`.
        """
        runs = sorted(self.runs.get(self.period, {}))
        self.param.run.objects = runs
        run = self.run if self.run in runs else (runs[0] if runs else None)
        self._set_quietly(run=run, index=0)
        if not defer:
            self._after_data_change()

    @_serialized
    def _first_render(self):
        """The initial event read, run once the page has been delivered.

        Everything the layout needs is already built by ``__init__``; only the
        data is outstanding. Deferring it means the user sees the dashboard
        (with a spinner on the array) instead of a blank tab while the first
        event is read. Outside a server context ``pn.state.onload`` fires
        immediately, so local runs and tests are unaffected.
        """
        try:
            self._after_data_change()
        finally:
            self.main_row.loading = False

    def _after_data_change(self):
        """Single render + dependent refreshes after a period/run/cycle change."""
        self._clear_selection()
        self._render()
        self._update_run_spectrum()
        self._update_validation()
        if self.playing:  # keep the playback end-stop in sync
            self._run_length = (
                self.viewer.run_length(self.period, self.run)
                if self.viewer is not None and self.period and self.run
                else None
            )

    @param.depends("production_cycle", watch=True)
    @_serialized
    def _on_cycle(self):
        self._load_cycle(self.production_cycle)
        self.run_spectrum = RunSpectrum(self.viewer)
        self.processor = WaveformProcessor(self.viewer)
        self.validation_data = validation.ValidationData(self.viewer)
        self._dataset_figs.clear()
        self._validation_figs.clear()
        self._cal_cache.clear()
        self._tab_state.clear()
        self._all_wf_figure = AllWaveformsFigure()
        self._update_dataset()
        periods = sorted(self.runs)
        self.param.period.objects = periods
        self._set_quietly(period=periods[0] if periods else None)
        self._apply_period()

    @param.depends("period", watch=True)
    @_serialized
    def _on_period(self):
        if self._internal_change:
            return
        self._apply_period()

    @param.depends("run", watch=True)
    @_serialized
    def _on_run(self):
        if self._internal_change:
            return
        # a new run starts at event 0 (the previous index is likely out of range)
        self._set_quietly(index=0)
        self._after_data_change()

    @param.depends("index", watch=True)
    @_serialized
    def _on_index(self):
        if self._internal_change:
            return
        self._render()

    @param.depends("waveform_param", "subtract_baseline", watch=True)
    @_serialized
    def _on_display_options(self):
        # display-only change: redraw the waveform panes, no event re-read
        self._update_waveforms()
        self._update_all_waveforms()

    def _render(self):
        if self.viewer is None:
            self.message.object = f"**production cycle:** {self._cycle_error}"
            self.message.visible = True
            return
        if not self.period or not self.run:
            return
        try:
            self.viewer.get_event(self.period, self.run, self.index)
            geds_data = build_source_data(self.viewer)
            sipm_data = sipm_view.build_source_data(self.viewer, geds_data)
            title = (
                f"{self.viewer.period} {self.viewer.run} {self.index} "
                f"- {self.viewer.event_timestamp}"
            )
            static_changed = self._source_tstamp != self.viewer.tstamp
            self._source_tstamp = self.viewer.tstamp

            def draw():
                # Only the energies change from event to event; the polygon
                # geometry and per-detector metadata are fixed by the
                # channelmap. Re-sending all of it every event costs ~80x the
                # bytes, and the ticker would rebuild a Bokeh model for labels
                # that did not move. _on_loop lands it all in one patch.
                if static_changed:
                    self.geds_source.data = geds_data
                    self.sipm_source.data = sipm_data
                    labels = sipm_view.row_label_map(geds_data)
                    self.figure.yaxis.ticker = FixedTicker(ticks=list(labels))
                    self.figure.yaxis.major_label_overrides = labels
                else:
                    _update_energy(self.geds_source, geds_data)
                    _update_energy(self.sipm_source, sipm_data)
                self.figure.title.text = title

            self._on_loop(draw)
            # the category refresh can re-point all_wf_category and fire its own
            # watcher; suspend it so the explicit rebuild below happens once
            with self._all_wf_batch():
                self._refresh_all_wf_categories()
            self._update_waveforms()
            self._update_spectrum()
            self._update_all_waveforms()
            self._update_event_details()
            self.message.visible = False
        except (IndexError, FileNotFoundError, KeyError, OSError) as exc:
            self.message.object = f"**{type(exc).__name__}:** {exc}"
            self.message.visible = True

    def _update_waveforms(self):
        if not self.show_waveforms or self.viewer is None or self.viewer.chmap is None:
            return
        self.wf_pane.object = plot_event_waveforms(
            self.viewer,
            param=self.waveform_param,
            processor=self.processor,
            subtract_baseline=self.subtract_baseline,
            selected=self.selected_detector,
        )

    def _update_spectrum(self):
        if (
            not self.show_spectrum
            or self.viewer is None
            or not self.period
            or not self.run
        ):
            return
        try:
            counts, edges = self._accumulated_histogram()
        except (KeyError, FileNotFoundError, OSError, IndexError) as exc:
            self.message.object = f"**spectrum:** {type(exc).__name__}: {exc}"
            self.message.visible = True
            return
        data = spectrum_view.source_data(counts, edges)
        self._on_loop(lambda: self.spectrum_source.update(data=data))

    def _accumulated_histogram(self):
        """Counts of every geds hit up to the current event.

        Playback walks the run one event at a time, so the common case is
        "same run, index advanced by one" -- add just that event's hits to the
        running counts instead of re-reducing the whole run every frame, which
        is what caps playback speed on a long run.
        """
        edges = np.histogram_bin_edges([], bins=N_BINS, range=ENERGY_RANGE)
        scope = (self.production_cycle, self.period, self.run)
        previous = self._accum_index
        if (
            scope == self._accum_scope
            and previous is not None
            and self.index > previous
        ):
            new = self.run_spectrum.hits_between(
                self.period, self.run, previous + 1, self.index
            )
            if new is not None:
                self._accum_counts += np.histogram(
                    new, bins=N_BINS, range=ENERGY_RANGE
                )[0]
                self._accum_index = self.index
                return self._accum_counts, edges

        counts, edges = self.run_spectrum.histogram(
            self.period, self.run, upto_index=self.index
        )
        self._accum_scope = scope
        self._accum_index = self.index
        self._accum_counts = counts
        return counts, edges

    def _update_event_details(self):
        if (
            self.tabs.active != TAB_DETAILS
            or self.viewer is None
            or self.viewer.chmap is None
        ):
            return
        key = (self.production_cycle, self.period, self.run, self.index)
        if self._tab_is_current(TAB_DETAILS, key):
            return
        tables = event_details.read_tables(self.viewer)
        self.summary_table.value = event_details.summary_dataframe(self.viewer, tables)
        for name, table in self.detail_tables.items():
            table.value = event_details.table_dataframe(tables, name)
        self._tab_drawn(TAB_DETAILS, key)

    def _cut_column(self, key):
        """A labelled column with the two checkboxes for one binary cut."""
        label, pos, neg = BINARY_CUTS[key]
        return pn.Column(
            pn.pane.Markdown(f"**{label}**", margin=(0, 5)),
            pn.widgets.Checkbox.from_param(self.param[f"cut_{key}_{pos}"], name=pos),
            pn.widgets.Checkbox.from_param(self.param[f"cut_{key}_{neg}"], name=neg),
        )

    def _cuts(self):
        cuts = {
            key: (
                getattr(self, f"cut_{key}_{pos}"),
                getattr(self, f"cut_{key}_{neg}"),
            )
            for key, (_label, pos, neg) in BINARY_CUTS.items()
        }
        cuts["multiplicity"] = self.cut_multiplicity
        return cuts

    def _update_run_spectrum(self):
        # whole-run, cut toggles; only when the Spectrum tab is active
        if (
            self.tabs.active != TAB_SPECTRUM
            or self.viewer is None
            or not self.period
            or not self.run
        ):
            return
        cuts = self._cuts()
        key = (
            self.production_cycle,
            self.period,
            self.run,
            self.spectrum_bin_width,
            tuple(sorted(cuts.items())),
        )
        if self._tab_is_current(TAB_SPECTRUM, key):
            return
        try:
            counts, edges = self.run_spectrum.histogram(
                self.period,
                self.run,
                cuts=cuts,
                bins=bins_for_width(self.spectrum_bin_width),
            )
        except (KeyError, FileNotFoundError, OSError, IndexError) as exc:
            self.message.object = f"**spectrum:** {type(exc).__name__}: {exc}"
            self.message.visible = True
            return
        data = spectrum_view.source_data(counts, edges)
        self._on_loop(lambda: self.run_spectrum_source.update(data=data))
        self._tab_drawn(TAB_SPECTRUM, key)

    @param.depends(
        "spectrum_bin_width",
        "cut_geds_trigger_forced",
        "cut_geds_trigger_normal",
        "cut_muon_coincident",
        "cut_muon_anticoincident",
        "cut_spms_coincident",
        "cut_spms_anticoincident",
        "cut_quality_pass",
        "cut_quality_fail",
        "cut_psd_pass",
        "cut_psd_fail",
        "cut_multiplicity",
        watch=True,
    )
    @_serialized
    def _on_spectrum_controls(self):
        self._clear_selection()  # bins (and their event sets) change
        self._update_run_spectrum()

    # -- spectrum-bin event selection -----------------------------------------

    def _clear_selection(self):
        self.event_selection = None
        self.selection_info.object = ""
        self._on_loop(lambda: self.run_spectrum_source.selected.update(indices=[]))

    def _on_bin_select(self, _attr, _old, new):
        self._dispatch(self._select_bin, new)

    @_serialized
    def _select_bin(self, new):
        if not new or self.viewer is None or not self.period or not self.run:
            self.event_selection = None
            self.selection_info.object = ""
            return
        data = self.run_spectrum_source.data
        lo, hi = float(data["left"][new[0]]), float(data["right"][new[0]])
        sel = self.run_spectrum.events_in_bin(
            self.period, self.run, self._cuts(), lo, hi
        )
        if len(sel) == 0:
            self._clear_selection()
            self.selection_info.object = "**Selection:** no events in this bin"
            return
        self.event_selection = sel
        self.selection_info.object = (
            f"**Selection:** {len(sel)} events in {lo:.0f}-{hi:.0f} keV"
        )
        self.index = int(sel[0])

    @param.depends("all_wf_system", watch=True)
    @_serialized
    def _on_all_wf_system(self):
        # switching system re-points grouping, kind and category; each has its
        # own watcher, so suspend them and rebuild once at the end -- otherwise
        # one dropdown change costs three ~100-channel reads
        with self._all_wf_batch():
            groupings = groupings_for(self.all_wf_system)
            self.param.all_wf_grouping.objects = groupings
            if self.all_wf_grouping not in groupings:
                self.all_wf_grouping = groupings[0]
            kinds = raw_kinds_for(self.all_wf_system)
            self.param.all_wf_kind.objects = kinds
            if self.all_wf_kind not in kinds:
                self.all_wf_kind = kinds[0]
            self._refresh_all_wf_categories()
        self._update_all_waveforms()

    @contextmanager
    def _all_wf_batch(self):
        """Coalesce the all-waveforms watchers into one rebuild by the caller."""
        self._suspend_all_wf = True
        try:
            yield
        finally:
            self._suspend_all_wf = False

    def _refresh_all_wf_categories(self):
        if self.viewer is None or self.viewer.chmap is None:
            return
        options = category_options(
            self.viewer, self.all_wf_system, self.all_wf_grouping
        )
        self.param.all_wf_category.objects = options
        if self.all_wf_category not in options:
            self.all_wf_category = "all"

    def _update_all_waveforms(self):
        # only the active "All waveforms" tab, to avoid 60 reads per playback step
        if (
            self._suspend_all_wf
            or self.tabs.active != TAB_WAVEFORMS
            or self.viewer is None
            or self.viewer.chmap is None
        ):
            return
        key = (
            self.production_cycle,
            self.period,
            self.run,
            self.index,
            self.all_wf_system,
            self.all_wf_grouping,
            self.all_wf_category,
            self.all_wf_exploded,
            self.all_wf_kind,
            self.waveform_param,
            self.subtract_baseline,
        )
        if self._tab_is_current(TAB_WAVEFORMS, key):
            return
        # a new event only refreshes the existing figure's data; the figure
        # (and the pane showing it) is replaced only when the layout changed,
        # so the browser keeps its renderers and its zoom
        self.all_wf_tab.loading = True
        try:
            rebuilt = self._all_wf_figure.update(
                self.viewer,
                system=self.all_wf_system,
                grouping=self.all_wf_grouping,
                category=self.all_wf_category,
                exploded=self.all_wf_exploded,
                param=self.waveform_param,
                processor=self.processor,
                subtract_baseline=self.subtract_baseline,
                kind=self.all_wf_kind,
                apply=self._on_loop,
            )
        finally:
            self.all_wf_tab.loading = False
        if rebuilt or self.all_wf_pane.object is not self._all_wf_figure.root:
            self.all_wf_pane.object = self._all_wf_figure.root
            # exploded subplots drop their y label in favour of one shared
            # label on the left; the compressed single figure keeps its own
            # and fills the row
            if self.all_wf_exploded:
                self.all_wf_ylabel.object = _vertical_label_html(
                    y_axis_label(self.waveform_param, self.subtract_baseline)
                )
                self.all_wf_area[:] = [self.all_wf_ylabel, self.all_wf_pane]
            else:
                self.all_wf_area[:] = [self.all_wf_pane]
        self._tab_drawn(TAB_WAVEFORMS, key)

    def _update_dataset(self):
        # metadata matrices are static per cycle, so build lazily and cache
        if self.tabs.active != TAB_DATASET or self.viewer is None:
            return
        # the "data" view shows all datatypes at once; the toggle is moot there
        all_datatypes = self.dataset_plot == "data"
        self.dataset_datatype_select.disabled = all_datatypes
        key = ("all" if all_datatypes else self.dataset_datatype, self.dataset_plot)
        if self._tab_is_current(TAB_DATASET, (self.production_cycle, key)):
            return  # already on screen; re-assigning re-serialises the figure
        fig = self._dataset_figs.get(key)
        if fig is None:
            self.dataset_tab.loading = True
            try:
                fig, source = dataset_view.dataset_figure(
                    self.viewer, plot=self.dataset_plot, datatype=self.dataset_datatype
                )
            except (KeyError, FileNotFoundError, OSError) as exc:
                self.message.object = f"**dataset view:** {type(exc).__name__}: {exc}"
                self.message.visible = True
                return
            finally:
                self.dataset_tab.loading = False
            # tap a cell/bar -> jump the event display to that run (handler
            # attached once per built figure; the cache prevents duplicates)
            source.selected.on_change(
                "indices", lambda _a, _o, new, s=source: self._on_dataset_tap(s, new)
            )
            self._dataset_figs[key] = fig
        # the user may have moved on while this built (under nthreads the tab
        # switch runs on another thread); the tab stays mounted, so the figure
        # is shown there and is ready when they return
        if fig is not self.dataset_pane.object:  # see _update_validation
            self.dataset_pane.object = fig
        self._tab_drawn(TAB_DATASET, (self.production_cycle, key))

    def _on_dataset_tap(self, source, new):
        self._dispatch(self._jump_to_run, source, new)

    @_serialized
    def _jump_to_run(self, source, new):
        if not new:
            return
        period, run = source.data["x"][new[0]]
        if run not in self.runs.get(period, {}):
            return  # metadata row without data in this cycle
        if (period, run) == (self.period, self.run):
            return
        self.param.run.objects = sorted(self.runs.get(period, {}))
        self._set_quietly(period=period, run=run, index=0)
        self._after_data_change()
        self.message.object = f"**dataset:** jumped to {period} {run} (event 0)"
        self.message.visible = True

    @param.depends("dataset_plot", "dataset_datatype", watch=True)
    @_serialized
    def _on_dataset_controls(self):
        self._update_dataset()

    def _update_validation(self):
        # everything is served from the binned per-run cache / parsed par
        # files, so rebuilding on control changes is cheap; figures are still
        # cached so revisiting a plot is instant
        if self.tabs.active != TAB_VALIDATION or self.viewer is None:
            return
        plot = self.validation_plot
        is_cal = plot.startswith("calibration")
        self.validation_bin_select.disabled = is_cal
        self.validation_log_toggle.disabled = is_cal
        self.validation_string_select.disabled = is_cal
        self.validation_detector_select.visible = plot == "calibration detail"
        if not is_cal:
            self._refresh_validation_strings()
        if self._tab_is_current(TAB_VALIDATION, self._validation_state()):
            return
        self.validation_tab.loading = True
        try:
            fig = self._validation_figure(plot)
        except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
            self.message.object = f"**validation:** {type(exc).__name__}: {exc}"
            self.message.visible = True
            return
        finally:
            self.validation_tab.loading = False
        if fig is not None:
            # never re-assign the figure already on screen: param treats any
            # object assignment as a change, Panel re-renders it, and its
            # ``_sync_properties`` then copies the rendered figure's themed
            # ``stylesheets`` (ImportedStyleSheet models) into the pane's
            # str-only param and raises (Panel 1.9)
            if fig is not self.validation_pane.object:
                self.validation_pane.object = fig
            # re-read the state: building a calibration plot can re-point
            # validation_detector at the run's first detector
            self._tab_drawn(TAB_VALIDATION, self._validation_state())

    def _validation_state(self):
        """Everything the validation pane is drawn from."""
        return (
            self.production_cycle,
            self.period,
            self.run,
            self.validation_plot,
            self.validation_bin_width,
            self.validation_log_y,
            self.validation_string,
            self.validation_detector,
        )

    def _validation_figure(self, plot):
        """Build (or fetch from cache) the requested validation figure."""
        if plot in validation_view.RATE_BUILDERS:
            scope = self.validation_string
            key = (
                self.period,
                plot,
                self.validation_bin_width,
                self.validation_log_y,
                scope,
            )
            fig = self._validation_figs.get(key)
            if fig is None:
                times, rates = self.validation_data.period_series(
                    self.period,
                    validation.BIN_WIDTHS[self.validation_bin_width],
                    string=None if scope == "all strings" else int(scope),
                )
                if times.size == 0:
                    self.message.object = (
                        f"**validation:** no event data in period {self.period}"
                    )
                    self.message.visible = True
                    return None
                fig = validation_view.RATE_BUILDERS[plot](
                    times,
                    rates,
                    self.validation_bin_width,
                    log_y=self.validation_log_y,
                    scope=scope if scope == "all strings" else f"string {scope}",
                )
            return self._cache_validation_fig(key, fig)

        # calibration plots: the run's residuals are ~100 numexpr evaluations
        # and are needed to populate the detector selector, so they are cached
        # per run -- otherwise merely picking a different detector recomputes
        # them all, even when the figure itself is already cached
        cal = self._cal_data()
        if cal is None:
            return None
        names, residuals, strings, label = cal
        self.param.validation_detector.objects = names
        if self.validation_detector not in names:
            self._set_quietly(validation_detector=names[0])

        if plot == "calibration summary":
            key = (self.period, self.run, plot)
            fig = self._validation_figs.get(key)
            if fig is None:
                fig = validation_view.cal_summary(names, residuals, strings, label)
        else:  # calibration detail
            key = (self.period, self.run, plot, self.validation_detector)
            fig = self._validation_figs.get(key)
            if fig is None:
                pars, _label = self.validation_data.load_cal_pars(self.period, self.run)
                curve = validation.cal_curve(pars, self.validation_detector)
                fig = validation_view.cal_detail(curve, self.validation_detector, label)
        return self._cache_validation_fig(key, fig)

    def _cal_data(self):
        """``(names, residuals, strings, label)`` for the run, cached.

        ``None`` (with the reason shown) when the run has no usable
        energy-calibration results.
        """
        key = (self.production_cycle, self.period, self.run)
        cached = self._cal_cache.pop(key, None)  # re-inserted below (LRU order)
        if cached is None:
            pars, label = self.validation_data.load_cal_pars(self.period, self.run)
            if pars is None:
                self.message.object = f"**validation:** {label}"
                self.message.visible = True
                return None
            names, residuals, strings = self._cal_residuals(pars)
            if not names:
                self.message.object = (
                    f"**validation:** no usable energy-calibration results in {label}"
                )
                self.message.visible = True
                return None
            cached = (names, residuals, strings, label)
        self._cal_cache[key] = cached
        while len(self._cal_cache) > 4:
            self._cal_cache.pop(next(iter(self._cal_cache)))
        return cached

    def _refresh_validation_strings(self):
        """Offer the period's strings in the scope selector."""
        opts = ["all strings"] + [
            str(s) for s in self.validation_data.available_strings(self.period)
        ]
        if self.param.validation_string.objects != opts:
            self.param.validation_string.objects = opts
            if self.validation_string not in opts:
                self._set_quietly(validation_string="all strings")

    def _cache_validation_fig(self, key, fig):
        self._validation_figs[key] = fig
        while len(self._validation_figs) > 16:
            self._validation_figs.pop(next(iter(self._validation_figs)))
        return fig

    def _cal_residuals(self, pars):
        """String-ordered residuals for the calibration summary plot."""
        try:
            ordered, string_nums = dataset_view._detector_rows(
                self.viewer, self.validation_data._start_key(self.period, self.run)
            )
            string_of = dict(zip(ordered, string_nums, strict=True))
        except Exception:  # no usable channelmap: fall back to par-file order
            ordered, string_of = sorted(pars), {}
        names, residuals = validation.cal_residuals(pars, ordered)
        return names, residuals, [string_of.get(n, -1) for n in names]

    @param.depends(
        "validation_plot",
        "validation_bin_width",
        "validation_log_y",
        "validation_string",
        "validation_detector",
        watch=True,
    )
    @_serialized
    def _on_validation_controls(self):
        if self._internal_change:
            return
        self._update_validation()

    @param.depends("all_wf_grouping", watch=True)
    @_serialized
    def _on_all_wf_grouping(self):
        self._refresh_all_wf_categories()
        self._update_all_waveforms()

    @param.depends("all_wf_category", "all_wf_exploded", "all_wf_kind", watch=True)
    @_serialized
    def _on_all_wf_controls(self):
        # label the toggle by the action it performs
        self.exploded_toggle.name = "Compressed" if self.all_wf_exploded else "Exploded"
        self._update_all_waveforms()

    @param.depends("show_waveforms", "show_spectrum", watch=True)
    @_serialized
    def _relayout(self):
        right = []
        if self.show_spectrum:
            right.append(self.spectrum_pane)
        if self.show_waveforms:
            right.append(self.wf_pane)
        if right:
            # fixed-width side column so the event array keeps most of the
            # width (and its aspect) instead of being squeezed into half
            self.main_row[:] = [
                self.bokeh_pane,
                pn.Column(*right, width=460, sizing_mode="stretch_height"),
            ]
        else:
            self.main_row[:] = [self.bokeh_pane]
        self._update_waveforms()
        self._update_spectrum()

    def _on_tap(self, _attr, _old, new):
        self._dispatch(self._select_detector, new)

    @_serialized
    def _select_detector(self, new):
        self.selected_detector = self.geds_source.data["name"][new[0]] if new else ""
        self._update_waveforms()

    @_serialized
    def _on_find(self, _event):
        if self.viewer is None:
            self.message.object = "**timestamp search:** no production cycle loaded"
            self.message.visible = True
            return
        try:
            target = parse_timestamp(self.timestamp_input.value)
            period, run, index = self.viewer.locate_timestamp(target)
        except (ValueError, FileNotFoundError, OSError) as exc:
            self.message.object = f"**timestamp search:** {exc}"
            self.message.visible = True
            return
        self.param.run.objects = sorted(self.runs.get(period, {}))
        self._set_quietly(period=period, run=run, index=index)
        self._after_data_change()

    @_serialized
    def _on_prev(self, _event):
        sel = self.event_selection
        if sel is not None:
            pos = int(np.searchsorted(sel, self.index, side="left")) - 1
            if pos >= 0:
                self.index = int(sel[pos])
        elif self.index > 0:
            self.index -= 1

    @_serialized
    def _on_next(self, _event):
        sel = self.event_selection
        if sel is not None:
            pos = int(np.searchsorted(sel, self.index, side="right"))
            if pos < len(sel):
                self.index = int(sel[pos])
        else:
            self.index += 1

    @param.depends("playing", "playback_period", watch=True)
    @_serialized
    def _playback(self):
        if self._playback_cb is not None:
            self._playback_cb.stop()
            self._playback_cb = None
        self.play_toggle.name = "⏸ Pause" if self.playing else "▶ Play run"
        if self.playing:
            self.show_spectrum = True  # surface the accumulating spectrum
            self._run_length = (
                self.viewer.run_length(self.period, self.run)
                if self.viewer is not None and self.period and self.run
                else None
            )
            self._playback_cb = pn.state.add_periodic_callback(
                self._advance, period=self.playback_period
            )

    def _advance(self):
        # the playback tick. Not _serialized: if the previous frame is still
        # rendering (a slow read, or the all-waveforms tab open), drop this
        # tick rather than queue frames behind it -- playback then runs as
        # fast as the data allows instead of falling ever further behind the
        # timer
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._run_length is None or self.index + 1 >= self._run_length:
                self.playing = False  # end of run (or no run loaded) -> stop
                return
            self.index += 1
        finally:
            self._lock.release()

    # -- layout ---------------------------------------------------------------

    def controls(self):
        selected = pn.bind(
            lambda d: f"**Selected detector:** {d or '—'}", self.param.selected_detector
        )
        return pn.Column(
            pn.widgets.Select.from_param(
                self.param.production_cycle, name="Production cycle"
            ),
            pn.layout.Divider(),
            pn.widgets.Select.from_param(self.param.period, name="Period"),
            pn.widgets.Select.from_param(self.param.run, name="Run"),
            pn.widgets.IntInput.from_param(self.param.index, name="Event index"),
            pn.Row(self.prev_button, self.next_button),
            self.selection_info,
            self.clear_button,
            self.timestamp_input,
            self.find_button,
            pn.Row(self.play_toggle),
            pn.widgets.IntSlider.from_param(
                self.param.playback_period,
                name="Playback interval (ms)",
                # each intermediate value would otherwise tear down and
                # re-register the periodic callback
                throttled=True,
            ),
            pn.widgets.Checkbox.from_param(
                self.param.show_spectrum, name="Show spectrum"
            ),
            pn.layout.Divider(),
            pn.widgets.Checkbox.from_param(
                self.param.show_waveforms, name="Show waveform panel"
            ),
            pn.widgets.Select.from_param(self.param.waveform_param, name="Waveform"),
            pn.widgets.Checkbox.from_param(
                self.param.subtract_baseline, name="Subtract baseline (raw only)"
            ),
            pn.pane.Markdown(selected),
            self.message,
        )

    def panel(self):
        return self.tabs


def create_app(base_path=None):
    """Build a fresh dashboard for one session (the Panel server entry point)."""
    display = EventDisplay(base_path)
    template = pn.template.FastListTemplate(
        title="LEGEND Event Display",
        logo=LEGEND_LOGO,
        favicon=LEGEND_FAVICON,
        header_background=HEADER_BACKGROUND,
        header_color=HEADER_COLOR,
        site_url="https://legend-exp.org",
        sidebar=[display.controls()],
        main=[display.panel()],
        sidebar_width=300,
    )
    header_items = [pn.HSpacer(), build_header_links()]
    chip = user_chip(pn.state.user)
    if chip is not None:
        # clear separation from the resource icons on the left and the
        # template's theme toggle on the right
        header_items += [
            pn.Spacer(width=40),
            pn.pane.HTML(chip, align="center", margin=(0, 20, 0, 0)),
        ]
    template.header.append(pn.Row(*header_items, sizing_mode="stretch_width"))
    return template
