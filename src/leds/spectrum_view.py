"""Bokeh histogram for the accumulating geds energy spectrum (log y)."""

from __future__ import annotations

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

LEGEND_BLUE = "#1A2A5B"
_BOTTOM = 0.5  # log axis floor: a count of 1 draws from here up to 1


def empty_source():
    return ColumnDataSource({"left": [], "right": [], "top": []})


def make_figure(source):
    fig = figure(
        sizing_mode="stretch_both",
        tools="tap,pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        x_axis_label="energy (keV)",
        y_axis_label="counts",
        y_axis_type="log",
        title="geds energy spectrum",
    )
    fig.quad(
        left="left",
        right="right",
        bottom=_BOTTOM,
        top="top",
        source=source,
        fill_color=LEGEND_BLUE,
        line_color=LEGEND_BLUE,
        alpha=0.85,
        selection_fill_color="#d62728",
        selection_line_color="#d62728",
        nonselection_fill_alpha=0.85,
    )
    return fig


def source_data(counts, edges):
    # only non-empty bins (a log axis cannot show zero-height bars)
    nz = counts > 0
    return {
        "left": edges[:-1][nz],
        "right": edges[1:][nz],
        "top": counts[nz],
    }
