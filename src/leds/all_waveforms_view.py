"""Bokeh "all waveforms" view, per detector system.

``system`` selects the detector system (geds / spms). ``grouping`` partitions
that system (see ``leds.event_viewer.SYSTEM_GROUPINGS``: geds by
String/HV filter/CC4, spms by barrel). ``category`` then picks the subset:

* ``"all"`` — every channel of the system,
* ``"above threshold"`` — geds channels above ``ABOVE_THRESHOLD`` (geds only),
* a specific group label (e.g. ``"String:01"``, ``"IB"``).

Views: *compressed* (one figure, traces overlaid + per-group/detector legend
with ``click_policy="hide"``) or *exploded* (a grid of subplots).

Traces are a raw waveform of the system's ``kind`` (geds:
waveform_windowed/presummed, spms: waveform_bit_drop), optionally
baseline-subtracted; for geds only, ``param`` may instead be a dsp-processed
waveform via a :class:`~leds.waveform_proc.WaveformProcessor`. Waveforms are
read at the event index (physics-mode tables are event-aligned).
"""

from __future__ import annotations

from bokeh.layouts import gridplot
from bokeh.models import Legend
from bokeh.palettes import Category20
from bokeh.plotting import figure
from lh5.io.exceptions import LH5DecodeError

from leds.event_viewer import SYSTEM_GROUPINGS, group_channels
from leds.waveform_view import RAW

ABOVE_THRESHOLD = 25.0  # keV
_READ_ERRORS = (OSError, KeyError, IndexError, LH5DecodeError)
SYSTEMS = tuple(SYSTEM_GROUPINGS)  # ("geds", "spms")
#: raw waveform kinds offered per system (geds also supports dsp via `param`)
SYSTEM_KINDS = {
    "geds": ("waveform_windowed", "waveform_presummed"),
    "spms": ("waveform_bit_drop",),
}


def groupings_for(system):
    return list(SYSTEM_GROUPINGS[system])


def raw_kinds_for(system):
    return list(SYSTEM_KINDS[system])


def category_options(ev, system, grouping):
    groups = list(group_channels(ev.chmap, system, grouping))
    # "above threshold" uses geds hit energies, so only for geds
    base = ["all", "above threshold"] if system == "geds" else ["all"]
    return base + groups


_MAX_LEGEND_ROWS = 16  # wrap into more columns beyond this many items


def _color(i):
    return Category20[20][i % 20]


def _legend(items, font_size="8pt"):
    ncols = max(1, -(-len(items) // _MAX_LEGEND_ROWS))  # ceil(n / max_rows)
    return Legend(
        items=items,
        click_policy="hide",
        label_text_font_size=font_size,
        ncols=ncols,
    )


def _trace(ev, rawid, name, param, processor, subtract_baseline, kind):
    if param == RAW:
        x, y = ev.read_waveform(rawid, ev.index, kind)
        y = y.astype(float)
        if subtract_baseline:
            y = y - y[:100].mean()
        return x, y
    return processor.processed(rawid, name, ev.index, param)


def y_axis_label(param, subtract_baseline):
    if param == RAW:
        return "amplitude (ADC)" + (
            ", baseline-subtracted" if subtract_baseline else ""
        )
    return param


def _detectors(ev, system, grouping, category):
    """Ordered ``[(rawid, name, group_label)]`` for the category."""
    groups = group_channels(ev.chmap, system, grouping)
    cmap = ev.chmap.map("daq.rawid")
    if category == "above threshold":
        fired = sorted(ev.fired_detectors, key=lambda d: d["energy"], reverse=True)
        return [
            (d["rawid"], d["name"], None)
            for d in fired
            if d["energy"] > ABOVE_THRESHOLD
        ]
    if category in groups:
        return [(r, cmap[r]["name"], category) for r in groups[category]]
    return [
        (r, cmap[r]["name"], label) for label, rawids in groups.items() for r in rawids
    ]


def _new_figure(param, subtract_baseline, **kwargs):
    return figure(
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        x_axis_label="time (ns)",
        y_axis_label=y_axis_label(param, subtract_baseline),
        **kwargs,
    )


def _ncols(n):
    if n <= 1:
        return 1
    if n <= 4:
        return 2
    if n <= 9:
        return 3
    return 4


def _compressed(
    ev, system, grouping, category, param, processor, subtract_baseline, kind
):
    fig = _new_figure(
        param, subtract_baseline, sizing_mode="stretch_both", title=category
    )
    dets = _detectors(ev, system, grouping, category)

    if category == "all":
        labels = list(group_channels(ev.chmap, system, grouping))
        color_of = {label: _color(i) for i, label in enumerate(labels)}
        grouped: dict = {}
        for rawid, name, group_label in dets:
            try:
                x, y = _trace(
                    ev, rawid, name, param, processor, subtract_baseline, kind
                )
            except _READ_ERRORS:
                continue
            line = fig.line(x, y, color=color_of[group_label], line_width=1, alpha=0.7)
            grouped.setdefault(group_label, []).append(line)
        items = [(label, grouped[label]) for label in labels if label in grouped]
    else:
        items = []
        for i, (rawid, name, _group) in enumerate(dets):
            try:
                x, y = _trace(
                    ev, rawid, name, param, processor, subtract_baseline, kind
                )
            except _READ_ERRORS:
                continue
            line = fig.line(x, y, color=_color(i), line_width=1.5)
            items.append((name, [line]))

    if items:
        fig.add_layout(_legend(items), "right")
    return fig


def _groups(ev, system, grouping, category):
    """``[(title, [(rawid, name), ...])]`` — one entry per exploded subplot."""
    groups = group_channels(ev.chmap, system, grouping)
    cmap = ev.chmap.map("daq.rawid")
    if category == "above threshold":
        fired = sorted(ev.fired_detectors, key=lambda d: d["energy"], reverse=True)
        return [
            (f"{d['name']} ({d['energy']:.0f} keV)", [(d["rawid"], d["name"])])
            for d in fired
            if d["energy"] > ABOVE_THRESHOLD
        ]
    if category in groups:
        return [(cmap[r]["name"], [(r, cmap[r]["name"])]) for r in groups[category]]
    return [
        (label, [(r, cmap[r]["name"]) for r in rawids])
        for label, rawids in groups.items()
    ]


def _exploded(
    ev, system, grouping, category, param, processor, subtract_baseline, kind
):
    groups = _groups(ev, system, grouping, category)
    figs = []
    shared_x = shared_y = None
    for title, members in groups:
        fig = _new_figure(
            param,
            subtract_baseline,
            height=240,
            sizing_mode="stretch_width",
            title=title,
        )
        fig.yaxis.axis_label = None  # one shared label for the whole grid instead
        if shared_x is None:
            shared_x, shared_y = fig.x_range, fig.y_range
        else:
            fig.x_range, fig.y_range = shared_x, shared_y
        items = []
        for j, (rawid, name) in enumerate(members):
            try:
                x, y = _trace(
                    ev, rawid, name, param, processor, subtract_baseline, kind
                )
            except _READ_ERRORS:
                continue
            items.append((name, [fig.line(x, y, color=_color(j), line_width=1)]))
        if items:
            fig.add_layout(_legend(items, font_size="7pt"), "right")
        figs.append(fig)

    if not figs:
        return _new_figure(
            param,
            subtract_baseline,
            sizing_mode="stretch_both",
            title="no detectors",
        )
    # stack the spms barrels (IB above OB) rather than side by side
    ncols = 1 if (system == "spms" and category == "all") else _ncols(len(figs))
    return gridplot(figs, ncols=ncols, sizing_mode="stretch_both")


def plot_all_waveforms(
    ev,
    *,
    system="geds",
    grouping="String",
    category="all",
    exploded=False,
    param=RAW,
    processor=None,
    subtract_baseline=True,
    kind="waveform_windowed",
):
    if system != "geds":
        param = RAW  # only geds has a dsp processing chain
    args = (ev, system, grouping, category, param, processor, subtract_baseline, kind)
    return _exploded(*args) if exploded else _compressed(*args)
