"""Data layer for the Validation tab: period-wide rate series + calibration.

Speed/memory design: the evt columns of one run are read once, immediately
reduced to time-binned counts (boost-histogram, 15-min base bins on a
day-aligned absolute axis) and dropped — only the tiny binned summaries are
cached (~150 kB per run), so a whole period never holds per-hit arrays.
Rebinning to the user-selected width is pure array arithmetic on the cache.
Every string's counts come from the same read, so scoping costs no re-read.

Calibration curves come from the ``par_hit`` / ``par_pht`` YAMLs valid at the
viewed run's start key (so ``validity.yaml`` decides which cal run applies),
not from lh5 data: the fitted peak centroids (ADC) and the calibration
expression are all that is needed, so only those parts are parsed.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path

import awkward as ak
import boost_histogram as bh
import h5py
import lh5
import numexpr as ne
import numpy as np
import yaml
from dbetto import AttrsDict, Props
from dbetto.catalog import Catalog
from lh5.io.exceptions import LH5DecodeError

from leds._cache import CAL_PARS, VALIDATION_SUMMARIES

#: Storage resolution of the per-run summaries. The UI bin widths below are
#: all multiples of this (and divide a day, so the day-aligned base axis
#: rebins exactly to any of them).
BASE_BIN_SECONDS = 900

#: UI label -> bin width in seconds.
BIN_WIDTHS = {
    "15 min": 900,
    "30 min": 1800,
    "1 h": 3600,
    "2 h": 7200,
    "3 h": 10800,
    "6 h": 21600,
    "12 h": 43200,
    "24 h": 86400,
}
DEFAULT_BIN_WIDTH = "1 h"

#: Potassium lines (keV) and the window (+- keV) counted around each.
K_LINES = {"K40": 1460.822, "K42": 1524.6}
LINE_WINDOW = 10.0

#: Peaks shown in the calibration-summary residual plot (keV).
PEAKS_SUMMARY = (2614.511, 583.191, 2103.511)

#: Calibration energy parameter whose curve is displayed (the production
#: energy estimator).
ENERGY_PARAM = "cuspEmax_ctc_cal"
#: Uncalibrated (ADC) input the calibration chain starts from.
RAW_ENERGY = ENERGY_PARAM.removesuffix("_cal")

#: ``results/ecal`` entries holding the ADC peak fits: ``par_hit`` stores them
#: under the final step, ``par_pht`` under its per-run step.
CAL_FITS = frozenset({ENERGY_PARAM, f"{RAW_ENERGY}_runcal"})

_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

#: evt fields read per group, one read per group; flags nested a level deeper
#: are read on their own.
EVT_FIELDS = {
    "trigger": ("timestamp", "is_forced"),
    "coincident": ("puls", "muon", "muon_offline", "spms"),
    "geds": ("multiplicity", "rawid", "energy"),
}
EVT_NESTED = ("geds/quality/is_bb_like", "geds/psd/is_bb_like")

#: Read only for the failing-flags plot, in their own pass, so the rate
#: summaries do not pay for them: the physics-event flags, and per
#: QC-failing hit its rawid and one bitmask per waveform class.
QC_FLAG_FIELDS = {
    "trigger": ("is_forced",),
    "coincident": ("puls",),
    "geds/quality/is_not_bb_like": (
        "rawid",
        "is_empty_bits",
        "is_pos_polarity_bits",
        "is_highly_pos_polarity_bits",
        "is_neg_polarity_bits",
        "is_delayed_discharge",
    ),
}

#: evt bitmask field -> the hit-tier candidate whose ``aggregations`` table
#: names its bits (bit set = that flag passed).
QC_CLASSES = {
    "is_empty_bits": "is_empty_candidate",
    "is_neg_polarity_bits": "is_negative_polarity_candidate",
    "is_pos_polarity_bits": "is_positive_polarity_candidate",
    "is_highly_pos_polarity_bits": "is_highly_positive_polarity_candidate",
}
QC_MAX_BITS = 16  # bits counted per class; the tables use at most 10

#: Calibration par tiers to look in, preferred first, per event tier:
#: partitioned event data (``pet``) is calibrated by the partition-level
#: ``par_pht``, per-run event data (``evt``) by ``par_hit``.
CAL_PAR_TIERS = {"pet": ("pht", "hit"), "evt": ("hit", "pht")}

#: ``validity.yaml`` categories tried in order. The viewed data is physics
#: (``Catalog`` falls back from ``phy`` to ``all`` by itself), but a catalog
#: may list its entries under ``cal`` only.
CAL_PAR_CATEGORIES = ("phy", "cal")

#: Cached per-channelmap string maps. A period usually spans a handful of
#: channelmap timestamps, and deriving one costs a channelmap build.
MAX_CACHED_STRING_MAPS = 8

_DAY = 86400

#: Series of each rate plot, in legend order. K-line groups are added below.
RATE_GROUPS = {
    "trigger": ("all triggers", "forced", "pulser", "muon", "muon offline"),
    "multiplicity": ("m = 0", "m = 1", "m = 2", "m > 2"),
    "qc": ("pass", "fail"),
}
KLINE_CONFIGS = ("before cuts", "after QC", "after mult = 1", "after LAr", "after PSD")
RATE_GROUPS |= dict.fromkeys(K_LINES, KLINE_CONFIGS)

#: Rate unit per group: seconds of exposure per rate unit (1 -> Hz,
#: 3600 -> counts/hour). All rates are additionally normalised by the
#: detector mass of the selected scope (one string, or the whole array).
GROUP_UNIT_SECONDS = {
    "trigger": 1,
    "multiplicity": 1,
    "qc": 1,
    **dict.fromkeys(K_LINES, 3600),
}
GROUP_UNIT_LABEL = {1: "rate (Hz / kg)", 3600: "rate (counts / hour / kg)"}

MULT_SELECTIONS = {
    "m = 0": lambda m: m == 0,
    "m = 1": lambda m: m == 1,
    "m = 2": lambda m: m == 2,
    "m > 2": lambda m: m > 2,
}


def _both(a, b):
    """``a & b``, or ``None`` if either is unavailable."""
    return a & b if a is not None and b is not None else None


def _not(a):
    return None if a is None else ~a


def _read_fields(group, handles, fields, prefix=""):
    """``{"<prefix>/<field>": lgdo or None}`` for ``fields`` of ``group``.

    One masked read of the group across the open files; if that fails,
    field by field, so a missing field only disables its own series.
    """
    keys = {f: f"{prefix}/{f}" if prefix else f for f in fields}
    if len(fields) > 1:
        try:
            tbl = lh5.read(group, handles, field_mask=list(fields))
            if tbl is not None and all(f in tbl for f in fields):
                return {keys[f]: tbl[f] for f in fields}
        except (KeyError, LH5DecodeError, OSError, ValueError):
            pass
    out = {}
    for f in fields:
        try:
            out[keys[f]] = lh5.read(f"{group}/{f}", handles)
        except (KeyError, LH5DecodeError, OSError):
            out[keys[f]] = None
    return out


def _read_groups(group, files, spec, nested=()):
    """``{"<sub>/<field>": lgdo or None}`` for every group in ``spec``.

    Each file is opened once for all of them, not once per field.
    """
    handles = []
    try:
        for f in files:
            handles.append(h5py.File(f, "r", locking=False))
        raw = {}
        for sub, fields in spec.items():
            raw |= _read_fields(f"{group}/{sub}", handles, fields, prefix=sub)
        for path in nested:
            raw |= _read_fields(group, handles, (path,))
        return raw
    finally:
        for h in handles:
            h.close()


def survival_fraction(pass_rate, fail_rate):
    """Per-bin fraction ``pass / (pass + fail)``, NaN where nothing was seen.

    The inputs are rate arrays over the same exposure, so their ratio equals
    the count ratio. Returns ``None`` if either series is unavailable.
    """
    if pass_rate is None or fail_rate is None:
        return None
    total = pass_rate + fail_rate
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, pass_rate / total, np.nan)


@functools.lru_cache(maxsize=16)
def qc_bit_tables(config_root):
    """``{evt bitmask field: [flag name per bit]}`` from a cycle's hit config.

    Read from the ``aggregations`` of the first ``tier/hit/*-hit_config.yaml``
    under ``config_root`` that defines all four QC candidates; ``{}`` if none.
    """
    for path in sorted(Path(config_root, "tier", "hit").glob("*-hit_config.yaml")):
        try:
            agg = yaml.load(path.read_text(), Loader=_YAML_LOADER).get("aggregations")
            tables = {
                evt: [agg[cand][f"bit{i}"] for i in range(len(agg[cand]))]
                for evt, cand in QC_CLASSES.items()
            }
        except (AttributeError, KeyError, TypeError, yaml.YAMLError, OSError):
            continue
        return tables
    return {}


def qc_flag_table(counts, tables):
    """Name the unset-bit counts of :meth:`ValidationData.period_qc_flags`.

    Returns ``(flags, {rawid: {flag: count}})``. A flag in several classes is
    the same stored boolean in each, so its first class is used. A class
    whose table is missing, or narrower than its stored bitmasks, is labelled
    ``"<class> bit N"`` instead.
    """
    flags, values = [], {}
    for name, per_rawid in counts["bits"].items():
        width = max(counts["max"].get(name, 0).bit_length(), 1)
        table = tables.get(name)
        if not table or len(table) < width:
            table = [f"{name.removesuffix('_bits')} bit {b}" for b in range(width)]
        for bit, flag in enumerate(table):
            if flag in flags:
                continue
            flags.append(flag)
            for rid, unset in per_rawid.items():
                values.setdefault(rid, {})[flag] = int(unset[bit])
    return flags, values


class ValidationData:
    """Per-run reduced rate summaries and their period-wide assembly."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._strings_cache: dict = {}

    # -- per-run reduction --------------------------------------------------

    def _columns(self, period, run):
        """Read the needed evt columns of one run (not cached: reduced at once).

        Each of the run's ~150 files is opened once for all fields, not once
        per field. Fields a cycle lacks come back as ``None``.
        """
        group = self.viewer.group
        files = self.viewer._run_files(period, run)
        if not files:
            return {"timestamp": np.empty(0)}  # _summary skips an empty run
        raw = _read_groups(group, files, EVT_FIELDS, EVT_NESTED)
        if raw["trigger/timestamp"] is None:
            msg = f"no {group}/trigger/timestamp in {period} {run}"
            raise KeyError(msg)

        def opt(field, conv):
            return None if raw[field] is None else conv(raw[field])

        as_bool = lambda o: o.nda.astype(bool)  # noqa: E731
        return {
            "timestamp": raw["trigger/timestamp"].nda,
            "forced": opt("trigger/is_forced", as_bool),
            "puls": opt("coincident/puls", as_bool),
            "muon": opt("coincident/muon", as_bool),
            "muon_offline": opt("coincident/muon_offline", as_bool),
            "spms": opt("coincident/spms", as_bool),
            "mult": opt("geds/multiplicity", lambda o: o.nda),
            "qc": opt("geds/quality/is_bb_like", as_bool),
            "rawid": opt("geds/rawid", lambda o: o.view_as("ak")),
            "energy": opt("geds/energy", lambda o: o.view_as("ak")),
            "psd_bb": opt(
                "geds/psd/is_bb_like",
                lambda o: ak.values_astype(o.view_as("ak"), bool),
            ),
        }

    @staticmethod
    def _cuts(d):
        """``(phys, cuts)``: the physics baseline and K-line cuts, per event.

        Forced-trigger and pulser events are not physics. ``cuts`` maps each
        K-line config but "after PSD" (a per-hit cut) to its event mask.
        ``None`` marks an unavailable field.
        """
        phys = _both(_not(d["forced"]), _not(d["puls"]))
        mult1 = None if d["mult"] is None else d["mult"] == 1
        return phys, {
            "before cuts": phys,
            "after QC": _both(phys, d["qc"]),
            "after mult = 1": _both(phys, mult1),
            "after LAr": _both(phys, _not(d["spms"])),
        }

    @classmethod
    def _series_masks(cls, d):
        """Per-event boolean masks for every rate series, ``None`` if unavailable.

        All series except the raw trigger components exclude non-physics
        events; the K-line configs apply one cut each on top of that.
        """
        phys, cuts = cls._cuts(d)
        masks = {
            ("trigger", "all triggers"): np.ones(len(d["timestamp"]), dtype=bool),
            ("trigger", "forced"): d["forced"],
            ("trigger", "pulser"): d["puls"],
            ("trigger", "muon"): d["muon"],
            ("trigger", "muon offline"): d["muon_offline"],
            ("qc", "pass"): _both(phys, d["qc"]),
            ("qc", "fail"): _both(phys, _not(d["qc"])),
        }
        for label, sel in MULT_SELECTIONS.items():
            cond = None if d["mult"] is None else sel(d["mult"])
            masks[("multiplicity", label)] = _both(phys, cond)

        energy, psd = d["energy"], d["psd_bb"]
        for line, peak in K_LINES.items():
            in_line = in_line_psd = None
            if energy is not None:
                win = (energy >= peak - LINE_WINDOW) & (energy < peak + LINE_WINDOW)
                in_line = ak.to_numpy(ak.any(win, axis=1))
                if psd is not None:
                    in_line_psd = ak.to_numpy(ak.any(win[psd], axis=1))
            for config, cut in cuts.items():
                masks[(line, config)] = _both(cut, in_line)
            masks[(line, "after PSD")] = _both(phys, in_line_psd)
        return masks

    @classmethod
    def _string_hists(cls, d, masks, axis, string_rawids):
        """Every series per string, as ``(time, string)`` histograms.

        An event counts toward a string when it has a hit there -- for the
        K-lines, an in-window hit there. One hit -> string lookup serves all
        strings, instead of re-deriving every mask per string. ``None``
        without ``geds/rawid``.
        """
        rawid = d.get("rawid")
        strings = sorted(string_rawids)
        lut = {r: i for i, s in enumerate(strings) for r in string_rawids[s]}
        if rawid is None or not lut:
            return None
        keys = np.array(sorted(lut), dtype=np.int64)
        flat = ak.to_numpy(ak.flatten(rawid)).astype(np.int64)
        pos = np.minimum(np.searchsorted(keys, flat), keys.size - 1)
        known = keys[pos] == flat  # hits of unmapped channels drop out
        hit_str = np.array([lut[k] for k in keys], dtype=np.int64)[pos]
        hit_evt = np.repeat(np.arange(len(rawid)), ak.to_numpy(ak.num(rawid)))
        t, values, ns = d["timestamp"], np.asarray(strings), len(strings)

        def pairs(hit_mask=None):
            # (event, string index) once per pair, from the selected hits
            sel = known if hit_mask is None else known & hit_mask
            pair = np.unique(hit_evt[sel] * ns + hit_str[sel])
            return pair // ns, pair % ns

        def fill(ev_st, evt_mask):
            if evt_mask is None:
                return None
            ev, st = ev_st
            keep = evt_mask[ev]
            h = bh.Histogram(axis, bh.axis.IntCategory(strings))
            h.fill(t[ev[keep]], values[st[keep]])
            return h

        any_hit = pairs()
        hists = {k: fill(any_hit, m) for k, m in masks.items() if k[0] not in K_LINES}
        phys, cuts = cls._cuts(d)
        energy, psd = d["energy"], d["psd_bb"]
        flat_e = flat_psd = None
        if energy is not None:
            flat_e = ak.to_numpy(ak.flatten(energy)).astype(float)
            if flat_e.size != flat.size:
                flat_e = None  # energy and rawid not hit-aligned: no K-lines
        if psd is not None and flat_e is not None:
            flat_psd = ak.to_numpy(ak.flatten(psd)).astype(bool)
        for line, peak in K_LINES.items():
            win = None
            if flat_e is not None:
                win = (flat_e >= peak - LINE_WINDOW) & (flat_e < peak + LINE_WINDOW)
            in_line = None if win is None else pairs(win)
            for config, cut in cuts.items():
                hists[(line, config)] = None if in_line is None else fill(in_line, cut)
            hists[(line, "after PSD")] = (
                None if flat_psd is None else fill(pairs(win & flat_psd), phys)
            )
        return hists

    def _string_map(self, period, run):
        """Per-string ged rawids and masses from the run's channelmap.

        Returns ``({string: frozenset(rawids)}, {string: mass_kg})``.
        """
        tstamps = self.viewer._run_tstamps(period, run)
        if not tstamps:
            msg = f"no data files for {period} {run}"
            raise ValueError(msg)
        cached = self._strings_cache.pop(tstamps[0], None)  # re-inserted below
        if cached is None:
            chmap = self.viewer._channelmap(tstamps[0])
            geds = chmap.map("system", unique=False).geds.map("name")
            rawids: dict = {}
            masses: dict = {}
            for name in geds:
                det = geds[name]
                string = int(det.location.string)
                rawids.setdefault(string, set()).add(int(det.daq.rawid))
                masses[string] = (
                    masses.get(string, 0.0) + float(det.production.mass_in_g) / 1000
                )
            cached = ({s: frozenset(v) for s, v in rawids.items()}, masses)
        self._strings_cache[tstamps[0]] = cached
        while len(self._strings_cache) > MAX_CACHED_STRING_MAPS:
            self._strings_cache.pop(next(iter(self._strings_cache)))
        return cached

    def _mass_kg(self, period, run, string=None):
        """Ged mass of the scope: one string, or (``None``) the whole array."""
        masses = self._string_map(period, run)[1]
        if string is None:
            return sum(masses.values())
        return masses.get(string, 0.0)

    def available_strings(self, period):
        """Sorted string numbers of the period's first readable channelmap."""
        for run in sorted(self.viewer.available_runs().get(period, {})):
            try:
                return sorted(self._string_map(period, run)[0])
            except Exception:  # no channelmap: the selector just offers "all"
                continue
        return []

    def _qc_flag_counts(self, period, run):
        """Per detector, how often each QC bit is unset, over physics events.

        Returns ``{"events": physics events, "failing": {rawid: failing hits},
        "bits": {class: {rawid: (QC_MAX_BITS,) unset counts}}, "max": {class:
        largest stored bitmask}, "discharge": delayed-discharge events}``, or
        ``None`` without the ``is_not_bb_like`` fields. Only QC-failing hits
        carry bitmasks. A pass of its own over the files, cached like
        :meth:`_summary`, so the rate plots never read these fields.
        """
        files = self.viewer._run_files(period, run)

        def build():
            raw = _read_groups(self.viewer.group, files, QC_FLAG_FIELDS)
            nbb = "geds/quality/is_not_bb_like"
            if raw.get(f"{nbb}/rawid") is None:
                return None
            rawid = raw[f"{nbb}/rawid"].view_as("ak")
            forced, puls = raw.get("trigger/is_forced"), raw.get("coincident/puls")
            keep = np.ones(len(rawid), dtype=bool)
            if forced is not None and puls is not None:
                keep = ~(forced.nda.astype(bool) | puls.nda.astype(bool))
            rid = ak.to_numpy(ak.flatten(rawid[keep])).astype(np.int64)
            ids, inverse, n_fail = np.unique(
                rid, return_inverse=True, return_counts=True
            )
            out = {
                "events": int(keep.sum()),
                "failing": dict(zip(ids.tolist(), n_fail.tolist(), strict=True)),
                "bits": {},
                "max": {},
                "discharge": None,
            }
            for name in QC_CLASSES:
                arr = raw.get(f"{nbb}/{name}")
                if arr is None:
                    continue
                flat = ak.to_numpy(ak.flatten(arr.view_as("ak")[keep]))
                if flat.size != rid.size:
                    continue  # not hit-aligned with the rawids
                # some cycles store a class as all-NaN floats: no information
                known = np.isfinite(flat) if flat.dtype.kind == "f" else slice(None)
                flat = flat[known].astype(np.int64)
                if flat.size == 0:
                    continue
                unset = ((flat[:, None] >> np.arange(QC_MAX_BITS)) & 1) == 0
                counts = np.stack(
                    [
                        np.bincount(inverse[known], unset[:, b], ids.size)
                        for b in range(QC_MAX_BITS)
                    ],
                    axis=1,
                ).astype(np.int64)
                out["bits"][name] = dict(zip(ids.tolist(), counts, strict=True))
                out["max"][name] = int(flat.max()) if flat.size else 0
            dd = raw.get(f"{nbb}/is_delayed_discharge")
            if dd is not None:
                out["discharge"] = int((dd.nda.astype(bool) & keep).sum())
            return out

        key = ("qc_flags", self.viewer.cycle_key, period, run, tuple(map(str, files)))
        return VALIDATION_SUMMARIES.get(key, build)

    def period_qc_flags(self, period):
        """:meth:`_qc_flag_counts` summed over every run of ``period``."""
        total = None
        for run in sorted(self.viewer.available_runs().get(period, {})):
            part = self._qc_flag_counts(period, run)
            if part is None:
                continue
            if total is None:
                total = {"events": 0, "failing": {}, "bits": {}, "max": {}}
                total["discharge"] = 0 if part["discharge"] is not None else None
            total["events"] += part["events"]
            for rid, n in part["failing"].items():
                total["failing"][rid] = total["failing"].get(rid, 0) + n
            for name, per in part["bits"].items():
                acc = total["bits"].setdefault(name, {})
                for rid, counts in per.items():
                    acc[rid] = acc.get(rid, 0) + counts
                total["max"][name] = max(total["max"].get(name, 0), part["max"][name])
            if part["discharge"] is not None and total["discharge"] is not None:
                total["discharge"] += part["discharge"]
        return total

    def _summary(self, period, run):
        """Binned counts of one run at base resolution, for every scope.

        ``series`` holds the whole-array histograms, ``strings`` the same
        series per string as ``(time, string)`` histograms (``None`` without
        ``geds/rawid``). Shared across sessions: an entry is ~2 MB but each is
        the reduction of a whole run's evt columns, so building one is the
        expensive part of a first visit to the Validation tab.
        """
        files = tuple(str(f) for f in self.viewer._run_files(period, run))

        def build():
            d = self._columns(period, run)
            t = d["timestamp"]
            if t.size == 0:
                return None
            # day-aligned absolute axis: every BIN_WIDTHS factor divides it
            # exactly, and bins of different runs line up with each other
            t0 = math.floor(t.min() / _DAY) * _DAY
            t1 = math.ceil(t.max() / _DAY) * _DAY
            t1 = max(t1, t0 + _DAY)
            axis = bh.axis.Regular(int((t1 - t0) / BASE_BIN_SECONDS), t0, t1)
            masks = self._series_masks(d)
            series = {}
            for skey, mask in masks.items():
                if mask is None:
                    series[skey] = None
                    continue
                h = bh.Histogram(axis)
                h.fill(t[mask])
                series[skey] = h
            strings = None
            if d.get("rawid") is not None:
                string_rawids = self._string_map(period, run)[0]
                strings = self._string_hists(d, masks, axis, string_rawids)
            # seconds of data coverage per bin (clipped overlap with the
            # run's [first, last] timestamp); gaps between the run's DAQ
            # cycles are not subtracted -- rates average over them
            edges = axis.edges
            exposure = bh.Histogram(axis, storage=bh.storage.Double())
            exposure.view()[:] = np.clip(
                np.minimum(edges[1:], t.max()) - np.maximum(edges[:-1], t.min()),
                0.0,
                None,
            )
            return {"series": series, "strings": strings, "exposure": exposure}

        key = (self.viewer.cycle_key, period, run, files)
        return VALIDATION_SUMMARIES.get(key, build)

    @staticmethod
    def _scope_counts(summary, key, string, factor):
        """Base counts of series ``key`` in scope, rebinned (``None`` if absent)."""
        if string is None:
            h = summary["series"][key]
            return None if h is None else h[:: bh.rebin(factor)].view()
        if summary["strings"] is None:
            msg = "this cycle's evt tier has no geds/rawid; cannot select a string"
            raise ValueError(msg)
        h = summary["strings"][key]
        if h is None:
            return None
        h = h[:: bh.rebin(factor), :]
        if string not in list(h.axes[1]):
            return np.zeros(h.axes[0].size)  # no such string in this run
        return h[:, bh.loc(string)].view()

    # -- period assembly ------------------------------------------------------

    def period_series(self, period, bin_seconds, string=None, keys=None):
        """Mass-normalised rates of every series over all runs of ``period``.

        Returns ``(times_ms, rates)`` where ``times_ms`` are bin centres in
        ms-epoch (Bokeh datetime axis) and ``rates`` maps ``(group, label)`` to
        an array aligned with ``times_ms`` (``None`` when the underlying field
        is absent in every run). Runs are separated by a NaN row so lines
        break across gaps. ``string`` restricts events to hits in that string;
        rates are divided by the scope's ged mass (per run, from its
        channelmap). Rebinned from the cached base-resolution summaries; no
        file is re-read when ``bin_seconds`` changes. ``keys`` limits the
        result to those series.
        """
        if bin_seconds < BASE_BIN_SECONDS or bin_seconds % BASE_BIN_SECONDS:
            msg = (
                f"bin_seconds must be a positive multiple of {BASE_BIN_SECONDS}, "
                f"got {bin_seconds}"
            )
            raise ValueError(msg)
        factor = bin_seconds // BASE_BIN_SECONDS
        all_keys = [(g, lbl) for g, labels in RATE_GROUPS.items() for lbl in labels]
        if keys is not None:
            all_keys = [k for k in all_keys if k in keys]
        times: list[np.ndarray] = []
        chunks: dict = {k: [] for k in all_keys}

        for run in sorted(self.viewer.available_runs().get(period, {})):
            summary = self._summary(period, run)
            if summary is None:
                continue
            mass = self._mass_kg(period, run, string)
            mass = mass if mass > 0 else np.nan
            exp = summary["exposure"][:: bh.rebin(factor)]
            exp_v = exp.view()
            covered = np.flatnonzero(exp_v > 0)
            if covered.size == 0:
                continue
            sl = slice(covered[0], covered[-1] + 1)
            centers = exp.axes[0].centers[sl]
            exp_v = exp_v[sl]
            if times:  # NaN separator between runs
                for k in all_keys:
                    chunks[k].append(np.array([np.nan]))
                times.append(np.array([(centers[0] - bin_seconds) * 1000.0]))
            times.append(centers * 1000.0)
            for k in all_keys:
                counts = self._scope_counts(summary, k, string, factor)
                if counts is None:
                    chunks[k].append(np.full(exp_v.size, np.nan))
                    continue
                counts = counts[sl].astype(float)
                unit = GROUP_UNIT_SECONDS[k[0]]
                chunks[k].append(counts / exp_v * unit / mass)

        if not times:
            return np.array([]), dict.fromkeys(all_keys)
        rates = {
            k: None if all(np.isnan(c).all() for c in parts) else np.concatenate(parts)
            for k, parts in chunks.items()
        }
        return np.concatenate(times), rates

    def ged_rows(self, period):
        """``[(label, rawid)]`` of the geds, in string order, for heatmap rows."""
        for run in sorted(self.viewer.available_runs().get(period, {})):
            tstamps = self.viewer._run_tstamps(period, run)
            if not tstamps:
                continue
            geds = self.viewer._channelmap(tstamps[0]).map("system", unique=False).geds
            dets = sorted(
                geds.map("name").values(),
                key=lambda d: (int(d.location.string), int(d.location.position)),
            )
            return [
                (f"s{int(d.location.string):02d} {d.name}", int(d.daq.rawid))
                for d in dets
            ]
        return []

    def qc_survival_by_string(self, period, bin_seconds):
        """``(times_ms, {string: survival fraction})`` over ``period``."""
        keys = [("qc", "pass"), ("qc", "fail")]
        times, fracs = np.array([]), {}
        for string in self.available_strings(period):
            times, rates = self.period_series(period, bin_seconds, string, keys)
            frac = survival_fraction(*(rates[k] for k in keys))
            if frac is not None:
                fracs[string] = frac
        return times, fracs

    # -- calibration parameters ------------------------------------------------

    def cal_par_sources(self, period, run):
        """Where the calibration pars valid for ``(period, run)`` are.

        Returns ``(sources, reason)``. ``sources`` lists ``(tier, root,
        category, start_key, files)`` for each par tier with pars valid at the
        run's start key, preferred tier first (see ``CAL_PAR_TIERS``); only
        the ``validity.yaml`` catalogs are read here. ``reason`` says what was
        searched, for when the list is empty.
        """
        start_key = self._start_key(period, run)
        if not start_key:
            return [], f"no start key for {period} {run}"
        tiers = CAL_PAR_TIERS.get(getattr(self.viewer, "tier", "evt"), ("hit", "pht"))
        sources, searched = [], []
        for tier in tiers:
            root = self.viewer.paths.get(f"par_{tier}")
            if not root:
                continue
            root = Path(root)
            validity = root / "validity.yaml"
            if not validity.is_file():
                searched.append(f"par_{tier}={root} (no validity.yaml)")
                continue
            files, category = self._valid_par_files(validity, start_key, tier)
            if not files:
                searched.append(f"par_{tier}={root} (no par_{tier} entry valid)")
                continue
            missing = [f for f in files if not (root / f).is_file()]
            if missing:
                searched.append(
                    f"par_{tier}={root} (listed file missing: {missing[0]})"
                )
                continue
            sources.append((tier, root, category, start_key, files))
        if sources:
            return sources, None
        where = (
            "; ".join(searched)
            or "this cycle's dataflow config has no par_hit/par_pht path"
        )
        return [], (
            f"no calibration pars valid at {start_key} for {period} {run} "
            f"(searched {where})"
        )

    @staticmethod
    def _valid_par_files(validity, start_key, tier):
        """``(files, category)`` of the ``par_<tier>`` YAMLs valid at ``start_key``."""
        try:
            catalog = Catalog.read_from(str(validity))
        except (ValueError, KeyError, TypeError):
            return [], None
        for category in CAL_PAR_CATEGORIES:
            try:
                entries = catalog.valid_for(start_key, category, allow_none=True)
            except ValueError:  # malformed start key
                return [], None
            files = [
                str(f) for f in (entries or []) if str(f).endswith(f"par_{tier}.yaml")
            ]
            if files:
                return files, category
        return [], None

    def load_cal_source(self, source):
        """``(pars, label)`` for one entry of :meth:`cal_par_sources`.

        The listed files are merged in order with dbetto's ``Props.add_to``,
        as ``TextDB.on`` does, but only their calibration part is parsed (see
        :func:`_read_cal_yaml`). Shared across sessions: every session
        looking at this run wants the same ones.
        """
        tier, root, _category, _start_key, files = source
        pars = load_par_files(root, files)
        # .../l200-p15-r005-cal-<ts>-par_hit.yaml -> "cal pars (hit): p15 r005"
        parts = Path(files[-1]).name.split("-")
        run = f"{parts[1]} {parts[2]}" if len(parts) > 2 else Path(files[-1]).name
        return pars, f"cal pars ({tier}): {run}"

    def load_cal_pars(self, period, run):
        """The calibration pars dict applying to ``(period, run)``.

        Returns ``(pars, source_label)`` from the preferred par tier, or
        ``(None, reason)`` when no tier has pars valid at the run's start key.
        """
        sources, reason = self.cal_par_sources(period, run)
        if not sources:
            return None, reason
        return self.load_cal_source(sources[0])

    def _start_key(self, period, run):
        runinfo = self.viewer.status_db.runinfo
        entry = runinfo.get(period, {}).get(run, {}).get("phy")
        if entry and entry.get("start_key"):
            return entry["start_key"]
        tstamps = self.viewer._run_tstamps(period, run)
        return tstamps[0] if tstamps else None


# -- par file reading ------------------------------------------------------------


def _slim_cal_yaml(text):
    """The lines of a par YAML that the calibration plots read.

    Keeps each detector's ``pars/operations`` and its ``results/ecal``
    entries in ``CAL_FITS``, relying on the dataflow's block-style dump
    (2-space indent, one key per line). That is 10-30% of a file.
    """
    out, path = [], [None] * 4  # detector, pars|results, section, entry
    for line in text.splitlines(keepends=True):
        body = line.lstrip(" ")
        if not body.strip():
            continue
        level = (len(line) - len(body)) // 2
        if level < 4 and not body.startswith("- ") and ":" in body:
            path[level] = body.split(":", 1)[0]
            path[level + 1 :] = [None] * (3 - level)
        top = (path[1], path[2])
        if (
            (level <= 1 and path[1] in (None, "pars", "results"))
            or (level == 2 and top in {("pars", "operations"), ("results", "ecal")})
            or (level >= 3 and top == ("pars", "operations"))
            or (level >= 3 and top == ("results", "ecal") and path[3] in CAL_FITS)
        ):
            out.append(line)
    return "".join(out)


def _has_fits(pars):
    """Whether any detector in ``pars`` has a ``CAL_FITS`` entry."""
    for det in pars.values():
        results = det.get("results") if isinstance(det, dict) else None
        ecal = results.get("ecal") if isinstance(results, dict) else None
        if isinstance(ecal, dict) and set(ecal) & CAL_FITS:
            return True
    return False


def _read_cal_yaml(path):
    """The calibration part of one par YAML (see :func:`_slim_cal_yaml`).

    Falls back to parsing the whole file when the slim parse fails or finds
    no peak fits, so a layout change costs speed, never the plot.
    """
    try:
        pars = yaml.load(_slim_cal_yaml(Path(path).read_text()), Loader=_YAML_LOADER)
        if isinstance(pars, dict) and _has_fits(pars):
            return pars
    except yaml.YAMLError:
        pass
    return Props.read_from(str(path))


def load_par_files(root, files):
    """The calibration part of par ``files`` (relative to ``root``), merged.

    Merged in order with dbetto's ``Props.add_to``, as ``TextDB.on`` does.
    Cached by the resolved files, so every run one validity entry covers
    shares one parse.
    """
    root = Path(root)

    def load():
        pars = AttrsDict()
        for f in files:
            pars = Props.add_to(pars, _read_cal_yaml(root / f))
        return pars

    return CAL_PARS.get((str(root), tuple(files)), load)


# -- calibration curve math (pure functions over a parsed par_hit/pht dict) ----


def _cal_chain(operations):
    """Steps ``(input, name, operation)`` from ``RAW_ENERGY`` to ``ENERGY_PARAM``.

    ``par_hit`` has one step. ``par_pht`` has two: a per-run
    ``cuspEmax_ctc_runcal`` of the ADC value, then the partition-level
    ``cuspEmax_ctc_cal`` of that. Raises ``KeyError`` for anything else.
    """
    chain, name = [], ENERGY_PARAM
    while len(chain) <= len(operations):
        op = operations[name]
        names = set(re.findall(r"[A-Za-z_]\w*", op["expression"]))
        inputs = [v for v in names if v == RAW_ENERGY or v in operations]
        if len(inputs) != 1 or inputs[0] == name:
            break
        chain.insert(0, (inputs[0], name, op))
        if inputs[0] == RAW_ENERGY:
            return chain
        name = inputs[0]
    msg = f"no single-input calibration chain from {RAW_ENERGY} to {ENERGY_PARAM}"
    raise KeyError(msg)


def _eval_cal(chain, x):
    """Apply a calibration ``chain`` (see :func:`_cal_chain`) to ADC value(s) ``x``."""
    out = np.asarray(x, dtype=float)
    for var, _name, op in chain:
        out = ne.evaluate(op["expression"], local_dict={var: out, **op["parameters"]})
    return np.asarray(out, dtype=float)


def cal_curve(pars, detector):
    """Calibration-curve data for one detector's detail plot.

    Returns a dict of aligned arrays: fitted peak centroids ``mu``/``mu_err``
    (ADC), their true energies ``peaks`` (keV), the calibrated positions
    ``cal_mu`` with propagated ``cal_err``, residuals, and a dense
    ``(line_x, line_y)`` sampling of the calibration expression.
    Raises ``KeyError`` if the detector has no usable ecal results.
    """
    det = pars[detector]
    chain = _cal_chain(det["pars"]["operations"])
    # the ADC peak fits are stored under the step that takes the ADC value
    pk_fits = det["results"]["ecal"][chain[0][1]]["pk_fits"]

    peaks, mus, mu_errs = [], [], []
    for peak_key, fit in sorted(pk_fits.items(), key=lambda kv: float(kv[0])):
        peak = float(peak_key)
        mu = fit.get("parameters", {}).get("mu")
        if not fit.get("validity") or mu is None or not np.isfinite(mu):
            continue
        peaks.append(peak)
        mus.append(float(mu))
        mu_errs.append(float(fit.get("uncertainties", {}).get("mu", np.nan)))
    if not peaks:
        msg = f"no valid peak fits for {detector}"
        raise KeyError(msg)

    peaks_arr = np.array(peaks)
    mus_arr = np.array(mus)
    mu_errs_arr = np.array(mu_errs)
    cal_mu = _eval_cal(chain, mus_arr)
    cal_err = np.abs(_eval_cal(chain, mus_arr + mu_errs_arr) - cal_mu)
    line_x = np.linspace(0.0, 1.1 * mus_arr.max(), 200)
    return {
        "peaks": peaks_arr,
        "mu": mus_arr,
        "mu_err": mu_errs_arr,
        "cal_mu": cal_mu,
        "cal_err": cal_err,
        "residual": cal_mu - peaks_arr,
        "line_x": line_x,
        "line_y": _eval_cal(chain, line_x),
        "expression": (
            chain[0][2]["expression"]
            if len(chain) == 1
            else "; ".join(f"{name} = {op['expression']}" for _, name, op in chain)
        ),
    }


def cal_residuals(pars, ordered_names):
    """Per-detector calibration residuals for the summary plot.

    ``ordered_names`` is the preferred (string/position) detector order; only
    detectors present in ``pars`` are kept, and par-file detectors missing
    from the ordering are appended. Returns ``(names, residuals)`` with
    ``residuals[peak] = (res, err)`` arrays aligned to ``names`` (NaN where a
    detector lacks a valid fit for that peak).
    """
    names = [n for n in ordered_names if n in pars]
    names += sorted(set(pars) - set(names))
    residuals = {
        peak: (np.full(len(names), np.nan), np.full(len(names), np.nan))
        for peak in PEAKS_SUMMARY
    }
    kept = []
    for name in names:
        try:
            curve = cal_curve(pars, name)
        except (KeyError, TypeError, AttributeError, ValueError):
            continue  # no usable ecal results for this detector
        kept.append(name)
        i = len(kept) - 1
        for peak in PEAKS_SUMMARY:
            (j,) = np.where(np.isclose(curve["peaks"], peak))
            if j.size:
                residuals[peak][0][i] = curve["residual"][j[0]]
                residuals[peak][1][i] = curve["cal_err"][j[0]]
    for peak in PEAKS_SUMMARY:
        residuals[peak] = (
            residuals[peak][0][: len(kept)],
            residuals[peak][1][: len(kept)],
        )
    return kept, residuals
