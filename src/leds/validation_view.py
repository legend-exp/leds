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
    Legend,
    LegendItem,
    LinearColorMapper,
    LogColorMapper,
    Span,
)
from bokeh.palettes import (
    Category10,
    Category20,
    OrRd9,
    RdBu11,
    Turbo256,
    Viridis256,
)
from bokeh.plotting import figure

from leds import calcheck
from leds.validation import (
    GROUP_UNIT_LABEL,
    GROUP_UNIT_SECONDS,
    K_LINES,
    QC_EVENTS,
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
    "qc failures by run",
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


def qc_survival(
    times_ms, rates, bin_label, log_y=False, scope="all strings", events="physics"
):
    """Fraction of ``events`` (see ``QC_EVENTS``) passing the quality cuts, per bin.

    A ratio of two same-scope rates, so the mass normalisation cancels; the
    string restriction still applies through the underlying series.
    """
    group, words = QC_EVENTS[events]
    frac = survival_fraction(rates.get((group, "pass")), rates.get((group, "fail")))
    fig = figure(
        x_axis_type="datetime",
        y_axis_type="log" if log_y else "linear",
        height=340,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"quality-cut survival fraction of {words} ({bin_label} bins, {scope})",
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


def qc_survival_by_string(times_ms, fracs, bin_label, events="physics"):
    """Quality-cut survival fraction, one line per string on one figure."""
    fig = figure(
        x_axis_type="datetime",
        height=380,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"quality-cut survival fraction per string, {QC_EVENTS[events][1]} "
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


def qc_failing_flags(rows, flags, table, counts, period, events="physics"):
    """Detectors x QC flags: how often each flag fails, per selected event.

    ``rows`` is ``[(label, rawid)]`` in string order; ``table`` maps rawid to
    ``{flag: failing hits it caused}`` (see :func:`leds.validation._qc_reasons`).
    """
    n_events = max(counts["events"], 1)
    columns = ["fails QC", *flags]
    xs, ys, frac, n = [], [], [], []
    for label, rid in rows:
        row = {"fails QC": counts["failing"].get(rid, 0), **table.get(rid, {})}
        for col in columns:
            xs.append(col)
            ys.append(label)
            c = row.get(col, 0)
            n.append(c)
            frac.append(c / n_events if c else np.nan)
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
        title=f"QC flags failing, per event, {period}, {QC_EVENTS[events][1]} "
        f"({counts['events']} events"
        + (f"; delayed discharge in {dd / n_events:.2%}" if dd is not None else "")
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


def _short_flag(flag):
    return flag.removeprefix("is_valid_").removeprefix("is_") if flag else "(no flag)"


def qc_failures_by_run(per_run, names, period, n=10, events="physics"):
    """Runs x rank: each run's ``n`` detectors failing QC most, and why.

    ``per_run`` is ``[(run, top)]`` with ``top`` from
    :func:`leds.validation.top_failures`; ``names`` maps rawid to a label
    (``"s04 V01240A"``). Each cell shows the detector, its leading flag and
    the fraction of physics events it fails QC in, coloured by that fraction.
    """
    runs = [run for run, _ in per_run]
    ranks = [f"#{k + 1}" for k in range(n)]
    xs, ys, det, sub, frac, text_color, tip = [], [], [], [], [], [], []
    fracs = [f for _, top in per_run for _, _, f, _, _ in top if f > 0]
    lo, hi = (min(fracs), max(fracs)) if fracs else (1e-6, 1.0)
    span = np.log(hi / lo) if hi > lo else 1.0
    for run, top in per_run:
        for k, (rid, failing, f, lead, share) in enumerate(top):
            label = names.get(rid, f"ch{rid}")
            string, _, name = label.partition(" ")
            xs.append(ranks[k])
            ys.append(run)
            det.append(f"{name or label} ({string})" if name else label)
            sub.append(f"{_short_flag(lead)} {f:.2%}")
            frac.append(f)
            dark = f > 0 and np.log(f / lo) / span > 0.55
            text_color.append("#FFFFFF" if dark else "#1B2530")
            share_t = f"{share:.0%} of its failing hits" if share == share else "--"
            tip.append(f"{failing} failing hits; leading flag in {share_t}")
    source = ColumnDataSource(
        {
            "x": xs,
            "y": ys,
            "det": det,
            "sub": sub,
            "frac": frac,
            "color": text_color,
            "tip": tip,
        }
    )
    mapper = LogColorMapper(palette=list(reversed(OrRd9)), low=lo, high=hi)
    fig = figure(
        x_range=ranks,
        y_range=list(reversed(runs)),
        height=44 * len(runs) + 110,
        sizing_mode="stretch_width",
        tools="save",
        toolbar_location="right",
        x_axis_location="above",
        title=f"detectors failing QC most, per run of {period}: leading flag and "
        f"fraction of {QC_EVENTS[events][1]}",
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
    for field, offset, size in (("det", 7, "9px"), ("sub", -7, "8px")):
        fig.text(
            "x",
            "y",
            text=field,
            source=source,
            text_align="center",
            text_baseline="middle",
            y_offset=offset,
            text_font_size=size,
            text_color="color",
        )
    fig.add_tools(
        HoverTool(
            tooltips=[
                ("run", "@y"),
                ("rank", "@x"),
                ("detector", "@det"),
                ("fails QC in", "@frac{0.000%} of the events"),
                ("", "@tip"),
            ]
        )
    )
    fig.add_layout(ColorBar(color_mapper=mapper, title="fraction", width=10), "right")
    fig.axis.major_label_text_font_size = "10px"
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    return fig


#: Shape match below which a cell is marked "?" (median is ~0.8).
MATCH_POOR = 0.3


def calibration_check(data, err, match, detector, detail, sections, progress):
    """Every detector x cal run (energy-scale error), then one detector's runs.

    ``data`` is :meth:`leds.calcheck.CalCheck.spectra` (uncut: the grid is
    always drawn without cuts); ``err``/``match`` the ``(n_det, n_run)``
    results of :func:`leds.calcheck.scale_match`; ``detector`` the
    drill-down's detector name; ``detail`` its
    :meth:`~leds.calcheck.CalCheck.detector_spectra` with ``sections``
    applied, or ``None`` for uncut; ``progress`` ``(uncut, section data,
    runs)`` built. Returns ``(layout, overview_source)``: a tap on a cell
    selects that detector, through the source's selection.
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
    uncut, cuts_ready, n = progress
    overview = figure(
        x_range=runs,
        y_range=list(reversed(labels)),
        height=14 * len(labels) + 120,
        sizing_mode="stretch_width",
        tools="tap,save",
        toolbar_location="right",
        x_axis_location="above",
        title=f"energy-scale error from the whole spectrum, no cuts "
        f"({uncut}/{n} cal runs ready); ? = poor match, … = building; "
        f"click a cell",
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

    # drill-down: the selected detector, every cal run, with the sections
    if detail is None:
        counts, row_ready = data["counts"][sel], ready[sel]
        cut = "no cuts"
    else:
        counts, row_ready = detail["counts"], detail["ready"]
        cut = (
            f"sections: {', '.join(sections)}; "
            f"{cuts_ready}/{n} cal runs with section data"
        )
    rows = np.array([calcheck.coarsen(counts[j]) for j in range(len(runs))])
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
        height=max(380, 22 * ((len(runs) + 1) // 2) + 60),
        sizing_mode="stretch_width",
        x_range=waterfall.x_range,
        y_axis_type="log",
        tools="xpan,xwheel_zoom,box_zoom,reset,save",
        toolbar_location="right",
        title=f"{labels[sel]}: one line per cal run ({cut}); "
        f"click a run in the legend to hide it",
    )
    items = []
    for j, run in enumerate(runs):
        if not row_ready[j]:
            continue
        line = lines.line(
            centers, np.maximum(rows[j], 0.5), line_color=palette[j], line_alpha=0.8
        )
        items.append(LegendItem(label=run, renderers=[line]))
    if items:
        legend = Legend(items=items, click_policy="hide", ncols=2)
        legend.label_text_font_size = "9px"
        lines.add_layout(legend, "right")
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
    errors = data["errors"] | ({} if detail is None else detail["errors"])
    if errors:
        notes.append(
            "unavailable: " + "; ".join(f"{r}: {e}" for r, e in errors.items())
        )
    if detail is not None and detail["missing"]:
        notes.append(f"not applied: {', '.join(sorted(detail['missing']))}")
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
