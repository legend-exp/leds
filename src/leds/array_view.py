"""Interactive Bokeh rendering of the detector-array event view.

Kept separate from :mod:`leds.event_viewer` (which stays framework-agnostic and
only produces data) so the Bokeh figure can be built, updated and tested on its
own. The Panel layer holds one :class:`~bokeh.models.ColumnDataSource` and
swaps its ``.data`` per event, so changing event mutates a few arrays instead
of re-rendering an image.
"""

from __future__ import annotations

import numpy as np
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from leds.event_viewer import build_strings_dict, get_plot_source
from leds.sipm_view import GREEN_PALETTE

# Colours matching the previous matplotlib semantics:
UNDER_COLOR = "#808080"  # working detector below threshold (energy < vmin)
NAN_COLOR = "#ffffff"  # detector with no value (not processable)

#: Columns held by the ColumnDataSource; listed once so empty/real data agree.
COLUMNS = ("xs", "ys", "name", "energy", "string", "position", "usability", "rawid")


def empty_source():
    return ColumnDataSource(data={c: [] for c in COLUMNS})


def build_source_data(ev):
    """Assemble the per-detector glyph columns for the current event."""
    channel_map = ev.chmap.map("daq.rawid")
    strings_dict = build_strings_dict(ev.chmap)
    xs, ys, rawids = get_plot_source(channel_map, strings_dict)

    names, energies, strings, positions, usabilities = [], [], [], [], []
    for rawid in rawids:
        det = channel_map[rawid]
        name = det["name"]
        value = ev.energy_dict.get(name)
        names.append(name)
        energies.append(np.nan if value is None else float(value))
        strings.append(det["location"]["string"])
        positions.append(det["location"]["position"])
        usabilities.append(det["analysis"]["usability"])

    return {
        "xs": [list(x) for x in xs],
        "ys": [list(y) for y in ys],
        "name": names,
        "energy": energies,
        "string": strings,
        "position": positions,
        "usability": usabilities,
        "rawid": list(rawids),
    }


def make_event_figure(geds_source, sipm_source, *, vmin=25, vmax=6000, spms_vmax=4):
    """Combined geds-array + SiPM figure; returns ``(fig, geds_glyph)``.

    The geds detectors are patches (viridis, keV) and the SiPMs are rectangles
    (green, p.e.) wrapped around them; tap selection acts on the geds glyph.
    """
    fig = figure(
        match_aspect=True,
        tools="tap,pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="right",
        x_axis_location=None,
        background_fill_color="white",
        sizing_mode="stretch_both",
    )
    fig.grid.visible = False
    fig.yaxis.major_tick_line_color = None
    fig.yaxis.minor_tick_line_color = None
    fig.yaxis.axis_line_color = None

    # geds detectors
    geds_mapper = LinearColorMapper(
        palette=Viridis256,
        low=vmin,
        high=vmax,
        low_color=UNDER_COLOR,
        nan_color=NAN_COLOR,
    )
    geds_glyph = fig.patches(
        xs="xs",
        ys="ys",
        source=geds_source,
        fill_color={"field": "energy", "transform": geds_mapper},
        line_color="black",
        line_width=0.5,
        nonselection_fill_alpha=1.0,
        nonselection_line_alpha=1.0,
        selection_line_color="red",
        selection_line_width=2.5,
    )
    fig.add_tools(
        HoverTool(
            renderers=[geds_glyph],
            tooltips=[
                ("detector", "@name"),
                ("energy", "@energy{0,0.0} keV"),
                ("string:pos", "@string:@position"),
                ("usability", "@usability"),
            ],
        )
    )
    fig.add_layout(
        ColorBar(color_mapper=geds_mapper, title="energy (keV)", title_standoff=10),
        "right",
    )

    # SiPMs
    spm_mapper = LinearColorMapper(
        palette=GREEN_PALETTE,
        low=1e-9,
        high=spms_vmax,
        low_color=UNDER_COLOR,
        nan_color=NAN_COLOR,
    )
    spm_glyph = fig.rect(
        x="x",
        y="y",
        width="w",
        height="h",
        source=sipm_source,
        fill_color={"field": "energy", "transform": spm_mapper},
        line_color="black",
        line_width=0.5,
    )
    fig.add_tools(
        HoverTool(
            renderers=[spm_glyph],
            tooltips=[
                ("SiPM", "@name"),
                ("energy", "@energy{0,0.00} p.e."),
                ("barrel:pos", "@barrel:@position"),
                ("fiber", "@fiber"),
            ],
        )
    )
    fig.add_layout(
        ColorBar(color_mapper=spm_mapper, title="p.e.", title_standoff=10), "right"
    )
    return fig, geds_glyph
