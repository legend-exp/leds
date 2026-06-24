"""Bokeh waveform view: the fired detectors' waveforms for one event.

Each detector is its own line renderer so the legend can toggle traces with
``click_policy="hide"`` (double-click to isolate) — the interactive
replacement for the old matplotlib ``pick_event`` handlers. The number of
traces varies per event, so the figure is rebuilt per event rather than
updating a fixed ColumnDataSource.

``param="raw"`` plots the raw windowed waveform straight from the raw tier; any
other ``param`` is a dsp-processed waveform obtained from a
:class:`~leds.waveform_proc.WaveformProcessor`.

A ``selected`` detector (clicked in the array) is drawn alongside the fired
detectors even if it is below threshold. Physics-mode tables are event-aligned,
so its waveform is read at the event index like any other.
"""

from __future__ import annotations

from bokeh.models import Legend
from bokeh.palettes import Category20
from bokeh.plotting import figure

RAW = "raw"
SELECTED_COLOR = "#000000"


def _color(i):
    return Category20[20][i % 20]


def _trace(ev, det, param, processor, subtract_baseline):
    """Return ``(x, y)`` for one detector's waveform in the chosen representation."""
    if param == RAW:
        x, y = ev.read_waveform(det["rawid"], det["hit_idx"])
        y = y.astype(float)
        if subtract_baseline:
            y = y - y[:100].mean()
        return x, y
    return processor.processed(det["rawid"], det["name"], det["hit_idx"], param)


def _selected_entry(ev, selected, fired_names):
    """Build a detector entry for the clicked detector, or None.

    Skipped when nothing is selected or the selection is already among the
    fired detectors (its trace is shown anyway). Uses the event index as the
    table row, valid because physics-mode tables are event-aligned.
    """
    if not selected or selected in fired_names:
        return None
    try:
        rawid = int(ev.chmap[selected].daq.rawid)
    except (KeyError, AttributeError):
        return None
    return {
        "name": selected,
        "rawid": rawid,
        "hit_idx": ev.index,
        "energy": ev.energy_dict.get(selected) or 0.0,
    }


def plot_event_waveforms(
    ev, *, param=RAW, processor=None, subtract_baseline=True, selected=None
):
    """Build a Bokeh figure of the current event's fired-detector waveforms.

    Detectors are drawn highest-energy first so the most relevant traces sit on
    top and at the top of the legend; a clicked detector is added in bold black.
    """
    if param == RAW:
        y_label = "amplitude (ADC)" + (
            ", baseline-subtracted" if subtract_baseline else ""
        )
    else:
        y_label = param

    fig = figure(
        sizing_mode="stretch_both",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        x_axis_label="time (ns)",
        y_axis_label=y_label,
    )

    detectors = sorted(ev.fired_detectors, key=lambda d: d["energy"], reverse=True)
    fired_names = {d["name"] for d in detectors}

    items = []
    for i, det in enumerate(detectors):
        x, y = _trace(ev, det, param, processor, subtract_baseline)
        line = fig.line(x, y, color=_color(i), line_width=1.5)
        items.append((f"{det['name']}  ({det['energy']:.0f} keV)", [line]))

    extra = _selected_entry(ev, selected, fired_names)
    if extra is not None:
        try:
            x, y = _trace(ev, extra, param, processor, subtract_baseline)
            line = fig.line(
                x, y, color=SELECTED_COLOR, line_width=2.5, line_dash="dashed"
            )
            items.insert(0, (f"★ {extra['name']}  ({extra['energy']:.0f} keV)", [line]))
        except (OSError, KeyError, IndexError):
            pass  # selection not readable (e.g. odd/auxiliary channel) -> skip

    if items:
        legend = Legend(items=items, click_policy="hide", label_text_font_size="9pt")
        fig.add_layout(legend, "right")
    else:
        fig.title.text = "no fired detectors in this event"

    return fig
