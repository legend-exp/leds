"""Bokeh views for the Validation tab: rate-vs-time series and calibration.

Rate figures share one builder (datetime x-axis, one line per series, NaN
gaps between runs, click-to-hide legend). The K-line view stacks a K40 and a
K42 figure with a linked x-range. The calibration views draw residuals vs
detector (summary) and one detector's ADC-to-keV curve (detail) from the data
prepared in :mod:`leds.validation`.

Framework-agnostic: builds Bokeh figures, no Panel.
"""

from __future__ import annotations

import numpy as np
from bokeh.layouts import column
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    LogColorMapper,
    Span,
)
from bokeh.palettes import Category10, Category20, RdBu11, Turbo256, Viridis256
from bokeh.plotting import figure

from leds import calcheck
from leds.validation import (
    GROUP_UNIT_LABEL,
    GROUP_UNIT_SECONDS,
    K_LINES,
    RATE_GROUPS,
    survival_fraction,
)

PLOTS = (
    "trigger rates",
    "multiplicity rates",
    "K-line rates",
    "qc survival",
    "calibration summary",
    "calibration detail",
    "qc survival by string",
    "qc failing flags",
    "calibration check",
)

LEGEND_BLUE = "#1A2A5B"
_PALETTE = Category10[10]

#: Peak colours in the calibration summary (matched to PEAKS_SUMMARY order).
_PEAK_COLORS = (LEGEND_BLUE, "#6BAE75", "#FF4444")


def _rates_figure(times_ms, rates, group, *, title, height=340, log_y=True):
    """One rate-vs-time figure for every available series of ``group``.

    ``log_y`` (user-toggleable) suits the decades the rates span (triggers vs
    muons, K lines across cuts); zero-rate bins have no log-scale position
    and are simply not drawn there.
    """
    fig = figure(
        x_axis_type="datetime",
        y_axis_type="log" if log_y else "linear",
        height=height,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=title,
    )
    fig.yaxis.axis_label = GROUP_UNIT_LABEL[GROUP_UNIT_SECONDS[group]]
    missing = []
    # one source holding the shared time axis and a column per series, rather
    # than one source (and one duplicated time array) each; likewise a single
    # HoverTool, since BokehJS hit-tests every tool on every mouse move
    columns = {"t": times_ms}
    points = []
    for i, label in enumerate(RATE_GROUPS[group]):
        series = rates.get((group, label))
        if series is None:
            missing.append(label)
            continue
        columns[label] = series
        points.append((label, _PALETTE[i % len(_PALETTE)]))

    source = ColumnDataSource(columns)
    for label, color in points:
        fig.line("t", label, source=source, color=color, legend_label=label)
        # the renderer name is what the shared hover reads back as $name
        fig.scatter(
            "t",
            label,
            source=source,
            color=color,
            size=4,
            legend_label=label,
            name=label,
        )
    if points:
        fig.add_tools(
            HoverTool(
                renderers=[r for r in fig.renderers if r.name],
                tooltips=[
                    ("series", "$name"),
                    ("time", "@t{%F %T}"),
                    ("rate", "@$name{0.000}"),
                ],
                formatters={"@t": "datetime"},
            )
        )
    if missing:
        fig.title.text += f"  (unavailable: {', '.join(missing)})"
    fig.legend.click_policy = "hide"
    fig.legend.location = "top_right"
    return fig


def trigger_rates(times_ms, rates, bin_label, log_y=True, scope="all strings"):
    return _rates_figure(
        times_ms,
        rates,
        "trigger",
        title=f"trigger rates ({bin_label} bins, {scope})",
        log_y=log_y,
    )


def multiplicity_rates(times_ms, rates, bin_label, log_y=True, scope="all strings"):
    return _rates_figure(
        times_ms,
        rates,
        "multiplicity",
        title=f"multiplicity rates, forced/pulser removed "
        f"({bin_label} bins, {scope})",
        log_y=log_y,
    )


def qc_survival(times_ms, rates, bin_label, log_y=False, scope="all strings"):
    """Fraction of physics events passing the quality cuts, per time bin.

    A ratio of two same-scope rates, so the mass normalisation cancels; the
    string restriction still applies through the underlying series.
    """
    frac = survival_fraction(rates.get(("qc", "pass")), rates.get(("qc", "fail")))
    fig = figure(
        x_axis_type="datetime",
        y_axis_type="log" if log_y else "linear",
        height=340,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"quality-cut survival fraction, forced/pulser removed "
        f"({bin_label} bins, {scope})",
        # a fixed 0..1 range only makes sense on a linear axis
        **({} if log_y else {"y_range": (0.0, 1.05)}),
    )
    fig.yaxis.axis_label = "survival fraction"
    if frac is None:
        fig.title.text += "  (quality flags unavailable)"
        return fig
    source = ColumnDataSource({"t": times_ms, "frac": frac})
    fig.line("t", "frac", source=source, color=LEGEND_BLUE)
    pts = fig.scatter("t", "frac", source=source, color=LEGEND_BLUE, size=4)
    fig.add_tools(
        HoverTool(
            renderers=[pts],
            tooltips=[("time", "@t{%F %T}"), ("survival", "@frac{0.0000}")],
            formatters={"@t": "datetime"},
        )
    )
    return fig


def kline_rates(times_ms, rates, bin_label, log_y=True, scope="all strings"):
    """K40 and K42 line rates, stacked with a linked x-range."""
    figs = []
    for line, peak in K_LINES.items():
        fig = _rates_figure(
            times_ms,
            rates,
            line,
            title=f"{line} line rate, {peak:.1f} keV, forced/pulser removed "
            f"({bin_label} bins, {scope})",
            height=280,
            log_y=log_y,
        )
        if figs:
            fig.x_range = figs[0].x_range
        figs.append(fig)
    return column(*figs, sizing_mode="stretch_width")


def qc_survival_by_string(times_ms, fracs, bin_label):
    """Quality-cut survival fraction, one line per string on one figure."""
    fig = figure(
        x_axis_type="datetime",
        height=380,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"quality-cut survival fraction per string, forced/pulser removed "
        f"({bin_label} bins; click a string to hide it)",
    )
    fig.yaxis.axis_label = "survival fraction"
    columns = {"t": times_ms} | {f"s{s}": f for s, f in fracs.items()}
    source = ColumnDataSource(columns)
    palette = Category20[20]
    for i, string in enumerate(fracs):
        color = palette[i % len(palette)]
        label = f"string {string}"
        fig.line("t", f"s{string}", source=source, color=color, legend_label=label)
        fig.scatter(
            "t",
            f"s{string}",
            source=source,
            color=color,
            size=3,
            legend_label=label,
            name=f"s{string}",
        )
    if fracs:
        fig.add_tools(
            HoverTool(
                renderers=[r for r in fig.renderers if r.name],
                tooltips=[
                    ("string", "$name"),
                    ("time", "@t{%F %T}"),
                    ("survival", "@$name{0.0000}"),
                ],
                formatters={"@t": "datetime"},
            )
        )
        fig.legend.click_policy = "hide"
        fig.legend.ncols = 2
        fig.add_layout(fig.legend[0], "right")
    else:
        fig.title.text += "  (no per-string quality flags)"
    return fig


def qc_failing_flags(rows, flags, table, counts, period):
    """Detectors x QC flags: how often each flag fails, per physics event.

    ``rows`` is ``[(label, rawid)]`` in string order; ``table`` maps rawid to
    ``{flag: unset count}`` over QC-failing hits (see
    :func:`leds.validation.qc_flag_table`).
    """
    events = max(counts["events"], 1)
    columns = ["fails QC", *flags]
    xs, ys, frac, n = [], [], [], []
    for label, rid in rows:
        row = {"fails QC": counts["failing"].get(rid, 0), **table.get(rid, {})}
        for col in columns:
            xs.append(col)
            ys.append(label)
            c = row.get(col, 0)
            n.append(c)
            frac.append(c / events if c else np.nan)
    positive = [f for f in frac if f == f]
    mapper = LogColorMapper(
        palette=Viridis256,
        low=min(positive, default=1e-6),
        high=max(positive, default=1.0),
        nan_color="#F2F2F2",
    )
    source = ColumnDataSource({"x": xs, "y": ys, "frac": frac, "n": n})
    labels = [label for label, _ in rows]
    dd = counts.get("discharge")
    fig = figure(
        x_range=columns,
        y_range=list(reversed(labels)),
        height=15 * len(labels) + 170,
        sizing_mode="stretch_width",
        tools="hover,save",
        toolbar_location="right",
        x_axis_location="above",
        title=f"QC flags failing, per physics event, {period} "
        f"({counts['events']} events"
        + (f"; delayed discharge in {dd / events:.2%}" if dd is not None else "")
        + ")",
        tooltips=[
            ("detector", "@y"),
            ("flag", "@x"),
            ("events", "@n"),
            ("fraction", "@frac{0.000%}"),
        ],
    )
    fig.rect(
        "x",
        "y",
        1,
        1,
        source=source,
        line_color="white",
        fill_color={"field": "frac", "transform": mapper},
    )
    fig.add_layout(ColorBar(color_mapper=mapper, title="fraction", width=10), "right")
    fig.xaxis.major_label_orientation = 0.9
    fig.axis.major_label_text_font_size = "9px"
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    return fig


#: Shape match below which a cell is marked "?" (median is ~0.8).
MATCH_POOR = 0.3


def calibration_check(data, err, match, detector, sections, progress):
    """Every detector x cal run (energy-scale error), then one detector's runs.

    ``data`` is :meth:`leds.calcheck.CalCheck.spectra`; ``err``/``match`` the
    ``(n_det, n_run)`` results of :func:`leds.calcheck.scale_match`;
    ``detector`` the drill-down's detector name; ``progress`` ``(uncut,
    section data, runs)`` built. Returns ``(layout, overview_source)``: a
    tap on a cell selects that detector, through the source's selection.
    """
    runs, dets, ready = data["runs"], data["dets"], data["ready"]
    labels = [d[0] for d in dets]
    names = [d[1] for d in dets]
    sel = names.index(detector) if detector in names else 0
    xs, ys, v, err_t, match_t, mark, det_i = [], [], [], [], [], [], []
    for i, label in enumerate(labels):
        for j, run in enumerate(runs):
            e, m = err[i, j], match[i, j]
            xs.append(run)
            ys.append(label)
            v.append(float(np.clip(e, -0.5, 0.5)) if e == e else np.nan)
            err_t.append(f"{e:+.2f} %" if e == e else "--")
            match_t.append(f"{m:.2f}" if m == m else "--")
            if not ready[i, j]:
                mark.append("…")
            elif e != e:
                mark.append("")
            else:
                mark.append("?" if m < MATCH_POOR else "")
            det_i.append(i)
    source = ColumnDataSource(
        {
            "x": xs,
            "y": ys,
            "v": v,
            "err": err_t,
            "match": match_t,
            "t": mark,
            "det": det_i,
        }
    )
    mapper = LinearColorMapper(
        palette=list(reversed(RdBu11)), low=-0.5, high=0.5, nan_color="#C9CED3"
    )
    cut = f"sections: {', '.join(sections)}" if sections else "no cuts"
    uncut, cuts_ready, n = progress
    state = f"{(cuts_ready if sections else uncut)}/{n} cal runs ready"
    overview = figure(
        x_range=runs,
        y_range=list(reversed(labels)),
        height=14 * len(labels) + 120,
        sizing_mode="stretch_width",
        tools="tap,save",
        toolbar_location="right",
        x_axis_location="above",
        title=f"energy-scale error from the whole spectrum ({cut}; {state}); "
        f"? = poor match, … = building; click a cell",
    )
    overview.rect(
        "x",
        "y",
        1,
        1,
        source=source,
        line_color="white",
        fill_color={"field": "v", "transform": mapper},
        nonselection_fill_alpha=1.0,
        selection_line_color="black",
        selection_line_width=2,
    )
    overview.text(
        "x",
        "y",
        text="t",
        source=source,
        text_align="center",
        text_baseline="middle",
        text_font_size="10px",
    )
    overview.rect(
        x=runs,
        y=[labels[sel]] * len(runs),
        width=1,
        height=1,
        fill_alpha=0,
        line_color="#222222",
        line_width=1.5,
    )
    overview.add_tools(
        HoverTool(
            tooltips=[
                ("detector", "@y"),
                ("cal run", "@x"),
                ("scale error", "@err"),
                ("match", "@match"),
            ]
        )
    )
    overview.add_layout(ColorBar(color_mapper=mapper, title="%", width=10), "right")
    overview.xaxis.major_label_orientation = 1.0
    overview.axis.major_label_text_font_size = "9px"
    overview.grid.grid_line_color = None
    overview.axis.axis_line_color = None

    # drill-down: the selected detector, every cal run
    rows = np.array(
        [calcheck.coarsen(data["counts"][sel, j]) for j in range(len(runs))]
    )
    logc = np.log10(1 + rows)
    peak = logc.max(axis=1, keepdims=True)
    image = np.divide(logc, peak, out=np.zeros_like(logc), where=peak > 0)
    e_max = calcheck.E_MAX
    waterfall = figure(
        height=13 * len(runs) + 100,
        sizing_mode="stretch_width",
        x_range=(0, e_max),
        y_range=(0, len(runs)),
        tools="xpan,xwheel_zoom,reset,save",
        toolbar_location="right",
        title=f"{labels[sel]}: every cal run (rows), log counts per row ({cut})",
    )
    waterfall.image(
        image=[image],
        x=0,
        y=0,
        dw=e_max,
        dh=len(runs),
        color_mapper=LinearColorMapper(palette=Viridis256, low=0, high=1),
    )
    waterfall.yaxis.ticker = [k + 0.5 for k in range(len(runs))]
    waterfall.yaxis.major_label_overrides = {k + 0.5: r for k, r in enumerate(runs)}
    waterfall.yaxis.major_label_text_font_size = "9px"
    waterfall.xaxis.axis_label = "energy (keV)"

    centers = (np.arange(rows.shape[1]) + 0.5) * calcheck.DISPLAY_BIN
    palette = [
        Turbo256[int(20 + 200 * k / max(len(runs) - 1, 1))] for k in range(len(runs))
    ]
    lines = figure(
        height=360,
        sizing_mode="stretch_width",
        x_range=waterfall.x_range,
        y_axis_type="log",
        tools="xpan,xwheel_zoom,box_zoom,reset,save",
        toolbar_location="right",
        title=f"{labels[sel]}: one line per cal run (colour = run order)",
    )
    lines.multi_line(
        xs=[centers] * len(runs),
        ys=[np.maximum(r, 0.5) for r in rows],
        line_color=palette,
        line_alpha=0.8,
    )
    lines.xaxis.axis_label = "energy (keV)"
    lines.yaxis.axis_label = f"counts / {calcheck.DISPLAY_BIN:g} keV"
    for fig in (waterfall, lines):
        for e in calcheck.TH_LINES:
            fig.add_layout(
                Span(
                    location=e,
                    dimension="height",
                    line_color="#888888",
                    line_dash="dotted",
                    line_alpha=0.8,
                )
            )
    notes = []
    if data["errors"]:
        notes.append(
            "unavailable: " + "; ".join(f"{r}: {e}" for r, e in data["errors"].items())
        )
    if data["missing"].get(labels[sel]):
        notes.append(
            f"not applied for {labels[sel]}: {', '.join(sorted(data['missing'][labels[sel]]))}"
        )
    if notes:
        lines.title.text += "  (" + " | ".join(notes) + ")"
    return column(overview, waterfall, lines, sizing_mode="stretch_width"), source


#: Rate-plot label -> builder(times_ms, rates, bin_label).
RATE_BUILDERS = {
    "trigger rates": trigger_rates,
    "multiplicity rates": multiplicity_rates,
    "K-line rates": kline_rates,
    "qc survival": qc_survival,
}


def cal_summary(names, residuals, strings, source_label):
    """Calibration residuals vs detector, one series per summary peak.

    ``strings`` maps each detector index to its string number (for boundary
    separators); pass ``None`` to skip them.
    """
    x = np.arange(len(names))
    fig = figure(
        height=380,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"calibration residuals per detector ({source_label})",
        x_range=(-1.0, len(names) + 0.5),
    )
    fig.yaxis.axis_label = "residual (keV)"
    fig.xaxis.ticker = x
    fig.xaxis.major_label_overrides = {int(i): n for i, n in zip(x, names, strict=True)}
    fig.xaxis.major_label_orientation = 1.2
    if strings is not None:
        for i in range(1, len(strings)):
            if strings[i] != strings[i - 1]:
                fig.add_layout(
                    Span(
                        location=i - 0.5,
                        dimension="height",
                        line_color="#BBBBBB",
                        line_dash="dashed",
                    )
                )
    for color, (peak, (res, err)) in zip(_PEAK_COLORS, residuals.items(), strict=True):
        source = ColumnDataSource(
            {
                "x": x,
                "res": res,
                "name": names,
                "err_xs": [[i, i] for i in x],
                "err_ys": [[r - e, r + e] for r, e in zip(res, err, strict=True)],
            }
        )
        label = f"{peak:.1f} keV"
        pts = fig.scatter(
            "x", "res", source=source, color=color, size=6, legend_label=label
        )
        fig.multi_line("err_xs", "err_ys", source=source, color=color, alpha=0.6)
        fig.add_tools(
            HoverTool(
                renderers=[pts],
                tooltips=[
                    ("detector", "@name"),
                    ("peak", label),
                    ("residual", "@res{0.000} keV"),
                ],
            )
        )
    fig.legend.click_policy = "hide"
    return fig


def cal_detail(curve, detector, source_label):
    """One detector's calibration curve (top) and residuals (bottom)."""
    top = figure(
        height=320,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"{detector} energy calibration ({source_label}): "
        f"{curve['expression']}",
    )
    top.xaxis.axis_label = "peak centroid (ADC)"
    top.yaxis.axis_label = "peak energy (keV)"
    top.line(curve["line_x"], curve["line_y"], color="#888888", line_dash="dashed")
    source = ColumnDataSource(
        {
            "mu": curve["mu"],
            "peak": curve["peaks"],
            "res": curve["residual"],
            "err": curve["cal_err"],
            "res_xs": [[p, p] for p in curve["peaks"]],
            "res_ys": [
                [r - e, r + e]
                for r, e in zip(curve["residual"], curve["cal_err"], strict=True)
            ],
        }
    )
    pts = top.scatter("mu", "peak", source=source, color=LEGEND_BLUE, size=8)
    top.add_tools(
        HoverTool(
            renderers=[pts],
            tooltips=[("peak", "@peak{0.000} keV"), ("centroid", "@mu{0.0} ADC")],
        )
    )

    bottom = figure(
        height=240,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title="residuals",
    )
    bottom.xaxis.axis_label = "peak energy (keV)"
    bottom.yaxis.axis_label = "calibrated - true (keV)"
    bottom.add_layout(Span(location=0, dimension="width", line_color="#BBBBBB"))
    res_pts = bottom.scatter("peak", "res", source=source, color=LEGEND_BLUE, size=8)
    bottom.multi_line("res_xs", "res_ys", source=source, color=LEGEND_BLUE, alpha=0.6)
    bottom.add_tools(
        HoverTool(
            renderers=[res_pts],
            tooltips=[
                ("peak", "@peak{0.000} keV"),
                ("residual", "@res{0.000} keV"),
                ("error", "@err{0.000} keV"),
            ],
        )
    )
    return column(top, bottom, sizing_mode="stretch_width")
