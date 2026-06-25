"""SiPM (spms) glyph data for the combined event view.

The SiPMs are drawn as rectangles in the *same* figure as the geds array,
wrapped around it vertically (outer barrel outermost):

    OB top
    IB top
    <geds array>
    IB bottom
    OB bottom

Rectangle positions are computed relative to the geds array's bounds so the two
detector systems share one coordinate space. Colour encodes the summed energy
of a channel's triggered coincident pulses on a green scale (saturating at
``vmax`` p.e.); no signal is grey and usability != "on" is white. The figure
itself is built in :func:`leds.array_view.make_event_figure`.
"""

from __future__ import annotations

import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.palettes import Greens256

UNDER_COLOR = "#808080"  # usable but no triggered signal (energy 0)
NAN_COLOR = "#ffffff"  # usability not "on"
# Greens256 runs dark -> light; reverse for light->dark and drop the near-white
# end so the low colours stay clearly green (white means "off").
GREEN_PALETTE = list(Greens256[::-1])[60:]

COLUMNS = ("x", "y", "w", "h", "name", "rawid", "barrel", "position", "fiber", "energy")

# Vertical bands relative to the geds array: +ve above (top), -ve below (bottom),
# magnitude 2 = outer barrel (outermost), 1 = inner barrel (next to the array).
_ROW_BANDS = {
    ("OB", "top"): 2,
    ("IB", "top"): 1,
    ("IB", "bottom"): -1,
    ("OB", "bottom"): -2,
}


def empty_source():
    return ColumnDataSource({c: [] for c in COLUMNS})


def _geom(geds_data):
    """Rectangle size and per-row y from the geds array bounds."""
    xs, ys = geds_data["xs"], geds_data["ys"]
    if not xs:
        return 0.0, 1.0, 0.8, 0.8, {}
    xmin, xmax = min(map(min, xs)), max(map(max, xs))
    ymin, ymax = min(map(min, ys)), max(map(max, ys))
    width = (xmax - xmin) or 1.0
    cx = (xmin + xmax) / 2
    pitch = width / 20  # outer barrel has 20 per row -> spans the array width
    rect_w = rect_h = pitch * 0.82
    band = pitch
    row_y = {
        key: (ymax + b * band if b > 0 else ymin + b * band)
        for key, b in _ROW_BANDS.items()
    }
    return cx, pitch, rect_w, rect_h, row_y


def _groups(chmap):
    """SiPMs grouped by (barrel, position), each sorted by fiber."""
    spms = chmap.map("system", unique=False).spms.map("name")
    groups: dict = {}
    for name in spms:
        loc = spms[name].location
        groups.setdefault((str(loc.barrel), str(loc.position)), []).append(
            (str(loc.fiber), name, int(spms[name].daq.rawid))
        )
    for items in groups.values():
        items.sort()
    return groups


def build_source_data(ev, geds_data):
    """Per-SiPM glyph columns, positioned around the geds array."""
    cx, pitch, rect_w, rect_h, row_y = _geom(geds_data)
    groups = _groups(ev.chmap)

    cols: dict = {c: [] for c in COLUMNS}
    for (barrel, position), y in row_y.items():
        items = groups.get((barrel, position), [])
        x0 = cx - (len(items) - 1) / 2 * pitch
        for i, (fiber, name, rawid) in enumerate(items):
            if ev.usability(name) != "on":
                energy = np.nan  # white
            else:
                energy = ev.spms_energy.get(rawid, 0.0)  # 0 -> grey
            cols["x"].append(x0 + i * pitch)
            cols["y"].append(y)
            cols["w"].append(rect_w)
            cols["h"].append(rect_h)
            cols["name"].append(name)
            cols["rawid"].append(rawid)
            cols["barrel"].append(barrel)
            cols["position"].append(position)
            cols["fiber"].append(fiber)
            cols["energy"].append(energy)
    return cols


def row_label_map(geds_data):
    """``{y: "OB top", ...}`` for labelling the SiPM rows on the y-axis."""
    *_, row_y = _geom(geds_data)
    return {y: f"{barrel} {position}" for (barrel, position), y in row_y.items()}
