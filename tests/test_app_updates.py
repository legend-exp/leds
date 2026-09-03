"""Guards against redundant work in :class:`EventDisplay`: figures rebuilt when
nothing they draw from changed, and callbacks of one session overlapping.

The expensive work behind each tab is one builder call -- refreshing the
all-waveforms figure reads ~100 waveforms, *rebuilding* it also makes the
browser redraw every renderer -- so counting *those* (rather than the
``_update_*`` methods, which self-gate and return early) measures what a
single user interaction actually costs. These are the regressions that are
otherwise invisible: the app stays correct, it just does the work three times.

Most tests need a real production cycle; set ``$LEDS_BASE_PATH`` (the mock
cycle used by the ``run-app`` skill is enough) or they skip.
"""

from __future__ import annotations

import inspect
import os
import threading
import time

import pytest

from leds import app as app_mod

#: Expensive work, as ``name: (holder, attribute)``. ``all_wf_build`` creates
#: the all-waveforms Bokeh models (what the browser then has to tear down and
#: redraw); ``all_wf_refresh`` only reads the waveforms into existing sources.
BUILDERS = {
    "plot_event_waveforms": (app_mod, "plot_event_waveforms"),
    "all_wf_build": (app_mod.AllWaveformsFigure, "_build"),
    "all_wf_refresh": (app_mod.AllWaveformsFigure, "_refresh"),
}


class Counter:
    """Patches the expensive builders and tallies real calls."""

    def __init__(self, monkeypatch):
        self.counts = dict.fromkeys(BUILDERS, 0)
        for name, (holder, attr) in BUILDERS.items():
            monkeypatch.setattr(holder, attr, self._wrap(name, getattr(holder, attr)))

    def _wrap(self, name, inner):
        def counted(*args, **kwargs):
            self.counts[name] += 1
            return inner(*args, **kwargs)

        return counted

    def reset(self):
        for key in self.counts:
            self.counts[key] = 0


@pytest.fixture
def display():
    if not os.environ.get("LEDS_BASE_PATH"):
        pytest.skip("set $LEDS_BASE_PATH to a production cycle to run these")
    disp = app_mod.EventDisplay()
    if disp.viewer is None or not disp.period:
        pytest.skip(f"no usable cycle: {disp._cycle_error}")
    return disp


@pytest.fixture
def counter(monkeypatch):
    return Counter(monkeypatch)


# ---------------------------------------------------------------- rebuild counts


def test_system_change_rebuilds_all_waveforms_once(display, counter):
    """Changing System must not fan out through the grouping/kind/category watchers."""
    display.tabs.active = app_mod.TAB_WAVEFORMS
    counter.reset()

    display.all_wf_system = "spms" if display.all_wf_system == "geds" else "geds"

    assert counter.counts["all_wf_build"] == 1
    assert counter.counts["all_wf_refresh"] == 1


def test_grouping_change_rebuilds_all_waveforms_once(display, counter):
    display.tabs.active = app_mod.TAB_WAVEFORMS
    counter.reset()

    groupings = display.param.all_wf_grouping.objects
    other = next((g for g in groupings if g != display.all_wf_grouping), None)
    if other is None:
        pytest.skip("only one grouping available for this system")
    display.all_wf_grouping = other

    assert counter.counts["all_wf_build"] == 1


def test_returning_to_a_tab_does_not_rebuild(display, counter):
    """Nothing changed while we were away, so there is nothing to redo."""
    display.tabs.active = app_mod.TAB_WAVEFORMS
    display.tabs.active = app_mod.TAB_EVENT
    counter.reset()

    display.tabs.active = app_mod.TAB_WAVEFORMS

    assert counter.counts["all_wf_build"] == 0
    assert counter.counts["all_wf_refresh"] == 0


def test_new_event_refreshes_the_active_tab_without_rebuilding(display, counter):
    """A new event refreshes the data once; the figure (and its zoom) survives."""
    display.tabs.active = app_mod.TAB_WAVEFORMS
    root = display._all_wf_figure.root
    counter.reset()

    display.index += 1

    assert counter.counts["all_wf_refresh"] == 1
    assert counter.counts["all_wf_build"] == 0
    assert display._all_wf_figure.root is root
    assert display.all_wf_pane.object is root


def test_inactive_tabs_are_not_refreshed_on_a_new_event(display, counter):
    """The gate is what keeps playback from reading ~100 waveforms per step."""
    display.tabs.active = app_mod.TAB_EVENT
    counter.reset()

    display.index += 1

    assert counter.counts["all_wf_refresh"] == 0
    assert counter.counts["all_wf_build"] == 0


# ---------------------------------------------------------------- tab switching


@pytest.mark.parametrize(
    ("tab", "pane"),
    [
        (app_mod.TAB_DATASET, "dataset_pane"),
        (app_mod.TAB_VALIDATION, "validation_pane"),
    ],
)
def test_returning_to_a_tab_does_not_touch_the_pane_object(display, tab, pane):
    """Re-assigning an already rendered figure trips Panel 1.9 (stylesheets TypeError)."""
    display.tabs.active = tab
    assert getattr(display, pane).object is not None, "the tab drew on first visit"
    events = []
    getattr(display, pane).param.watch(events.append, "object")

    display.tabs.active = app_mod.TAB_EVENT
    display.tabs.active = tab

    assert events == []


def test_a_failing_updater_does_not_abort_the_others(display, monkeypatch):
    """One tab's bug must surface as a message, not leave another tab half-switched."""
    calls = []

    def boom():
        msg = "synthetic dataset failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(display, "_update_dataset", boom)
    monkeypatch.setattr(
        display, "_update_validation", lambda: calls.append("validation")
    )

    display.tabs.active = app_mod.TAB_VALIDATION

    assert calls == ["validation"], "the updater after the failing one still ran"
    assert display.message.visible
    assert "RuntimeError" in display.message.object
    assert "dataset" in display.message.object


def test_a_build_finished_after_leaving_the_tab_is_ready_on_return(
    display, monkeypatch
):
    """Under nthreads the user can switch away while a tab still builds.

    Tabs are static (always mounted), so the finished figure goes into the
    hidden tab and the return costs nothing: no rebuild, no re-assignment.
    """
    original = app_mod.dataset_view.dataset_figure
    built = []

    def build_then_leave(*args, **kwargs):
        result = original(*args, **kwargs)
        built.append(1)
        display.tabs.active = app_mod.TAB_EVENT  # the user has moved on meanwhile
        return result

    monkeypatch.setattr(app_mod.dataset_view, "dataset_figure", build_then_leave)

    display.tabs.active = app_mod.TAB_DATASET

    assert built == [1]
    assert display.dataset_pane.object is not None, "shown in the hidden tab"
    events = []
    display.dataset_pane.param.watch(events.append, "object")

    display.tabs.active = app_mod.TAB_DATASET

    assert built == [1], "not rebuilt"
    assert events == [], "not re-assigned"


def test_tabs_are_static():
    """Dynamic tabs re-render on the server per switch; rapid switching blanked them."""
    if not os.environ.get("LEDS_BASE_PATH"):
        pytest.skip("set $LEDS_BASE_PATH to a production cycle to run these")
    disp = app_mod.EventDisplay()
    assert disp.tabs.dynamic is False


def test_slow_tabs_show_loading_while_building(display, monkeypatch):
    original = app_mod.dataset_view.dataset_figure
    seen = []

    def observe(*args, **kwargs):
        seen.append(display.dataset_tab.loading)
        return original(*args, **kwargs)

    monkeypatch.setattr(app_mod.dataset_view, "dataset_figure", observe)

    display.tabs.active = app_mod.TAB_DATASET

    assert seen == [True], "loading while the figure builds"
    assert display.dataset_tab.loading is False


# ---------------------------------------------------------------- serialisation

#: Entry points that are not param watchers: Panel button/tab callbacks, the
#: onload hook, and the bodies the Bokeh ``on_change`` handlers dispatch to.
ENTRY_POINTS = (
    "_on_tab",
    "_first_render",
    "_on_find",
    "_on_prev",
    "_on_next",
    "_on_clear",
    "_select_detector",
    "_select_bin",
    "_jump_to_run",
)


def test_every_callback_entry_point_is_serialized():
    """A watcher added without the lock is a race under nthreads; catch it here."""
    members = vars(app_mod.EventDisplay)
    watchers = [
        name
        for name, fn in members.items()
        if inspect.isfunction(fn) and hasattr(fn, "_dinfo")  # param.depends marker
    ]
    assert len(watchers) >= 10, "param.depends no longer marks methods this way?"

    unguarded = [
        name
        for name in (*watchers, *ENTRY_POINTS)
        if not getattr(members[name], "_serialized", False)
    ]
    assert unguarded == []


def test_callbacks_of_one_session_never_overlap(display, monkeypatch):
    display.tabs.active = app_mod.TAB_WAVEFORMS
    monkeypatch.setattr(display, "_tab_is_current", lambda *_a: False)  # always redraw
    in_flight, peak, guard = [0], [0], threading.Lock()
    original = app_mod.AllWaveformsFigure._refresh

    def slow_refresh(self, *args, **kwargs):
        with guard:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.05)
        try:
            return original(self, *args, **kwargs)
        finally:
            with guard:
                in_flight[0] -= 1

    monkeypatch.setattr(app_mod.AllWaveformsFigure, "_refresh", slow_refresh)

    threads = [threading.Thread(target=display._on_index) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert peak[0] == 1


def test_playback_tick_is_dropped_while_the_session_is_busy(display):
    display._run_length = 10
    start = display.index
    acquired, release = threading.Event(), threading.Event()

    def busy():
        with display._lock:
            acquired.set()
            release.wait(timeout=5)

    other = threading.Thread(target=busy)
    other.start()
    assert acquired.wait(timeout=5)

    display._advance()  # another callback of this session is mid-flight
    assert display.index == start, "dropped, not queued"

    release.set()
    other.join(timeout=5)
    display._advance()
    assert display.index == start + 1
