"""Data layer for the Validation tab: period-wide rate series + calibration.

Speed/memory design: the evt columns of one run are read once, immediately
reduced to time-binned counts (boost-histogram, 15-min base bins on a
day-aligned absolute axis) and dropped — only the tiny binned summaries are
cached (~150 kB per run), so a whole period never holds per-hit arrays.
Rebinning to the user-selected width is pure array arithmetic on the cache.

Calibration curves come from the ``par_hit`` / ``par_pht`` YAMLs, loaded
through dbetto's ``TextDB`` at the viewed run's start key (so ``validity.yaml``
decides which cal run applies), not from lh5 data: the fitted peak centroids
(ADC) and the calibration expression are all that is needed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import awkward as ak
import boost_histogram as bh
import lh5
import numexpr as ne
import numpy as np
from dbetto import TextDB
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


class ValidationData:
    """Per-run reduced rate summaries and their period-wide assembly."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._strings_cache: dict = {}

    # -- per-run reduction --------------------------------------------------

    def _columns(self, period, run):
        """Read the needed evt columns of one run (not cached: reduced at once)."""
        files = [str(f) for f in self.viewer._run_files(period, run)]
        group = self.viewer.group

        def col(field):
            return lh5.read(f"{group}/{field}", files)

        def opt(field, conv):
            # tolerate cycles without a field: the series is just absent
            try:
                return conv(col(field))
            except (KeyError, LH5DecodeError, OSError):
                return None

        as_bool = lambda o: o.nda.astype(bool)  # noqa: E731
        return {
            "timestamp": col("trigger/timestamp").nda,  # required
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
    def _series_masks(d, hit_sel=None):
        """Per-event boolean masks for every rate series, ``None`` if unavailable.

        All series except the raw trigger components exclude forced-trigger and
        pulser events (they are not physics); the K-line configs additionally
        apply one cut each on top of that baseline. ``hit_sel`` (a per-hit
        boolean array, e.g. hits in one string) restricts every series to
        events with a selected hit, and the K-line windows to the selected
        hits themselves.
        """
        n = len(d["timestamp"])
        true = np.ones(n, dtype=bool)
        has_hit = None if hit_sel is None else ak.to_numpy(ak.any(hit_sel, axis=1))

        def both(a, b):
            return a & b if a is not None and b is not None else None

        def scoped(m):
            return m if has_hit is None else both(m, has_hit)

        phys = (
            ~(d["forced"] | d["puls"])
            if d["forced"] is not None and d["puls"] is not None
            else None
        )
        masks = {
            ("trigger", "all triggers"): scoped(true),
            ("trigger", "forced"): scoped(d["forced"]),
            ("trigger", "pulser"): scoped(d["puls"]),
            ("trigger", "muon"): scoped(d["muon"]),
            ("trigger", "muon offline"): scoped(d["muon_offline"]),
            ("qc", "pass"): scoped(both(phys, d["qc"])),
            ("qc", "fail"): scoped(
                both(phys, ~d["qc"] if d["qc"] is not None else None)
            ),
        }
        for label, sel in (
            ("m = 0", lambda m: m == 0),
            ("m = 1", lambda m: m == 1),
            ("m = 2", lambda m: m == 2),
            ("m > 2", lambda m: m > 2),
        ):
            cond = sel(d["mult"]) if d["mult"] is not None else None
            masks[("multiplicity", label)] = scoped(both(phys, cond))

        for line, peak in K_LINES.items():
            lo, hi = peak - LINE_WINDOW, peak + LINE_WINDOW
            if d["energy"] is None:
                in_line = in_line_psd = None
            else:
                # the in-window hit must itself be a selected (in-string) hit,
                # so these masks need no extra has_hit scoping
                e = d["energy"] if hit_sel is None else d["energy"][hit_sel]
                in_line = ak.to_numpy(ak.any((e >= lo) & (e < hi), axis=1))
                if d["psd_bb"] is None:
                    in_line_psd = None
                else:
                    psd = d["psd_bb"] if hit_sel is None else d["psd_bb"][hit_sel]
                    e_psd = e[psd]
                    in_line_psd = ak.to_numpy(
                        ak.any((e_psd >= lo) & (e_psd < hi), axis=1)
                    )
            base = both(phys, in_line)
            masks[(line, "before cuts")] = base
            masks[(line, "after QC")] = both(base, d["qc"])
            masks[(line, "after mult = 1")] = both(
                base, d["mult"] == 1 if d["mult"] is not None else None
            )
            masks[(line, "after LAr")] = both(
                base, ~d["spms"] if d["spms"] is not None else None
            )
            masks[(line, "after PSD")] = both(phys, in_line_psd)
        return masks

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

    def _hit_selection(self, d, period, run, string):
        """Per-hit boolean mask marking hits in ``string`` (ak, VoV layout)."""
        if d["rawid"] is None:
            msg = "this cycle's evt tier has no geds/rawid; cannot select a string"
            raise ValueError(msg)
        rawids = self._string_map(period, run)[0].get(string, frozenset())
        flat = np.isin(
            ak.flatten(d["rawid"]).to_numpy(),
            np.fromiter(rawids, dtype=np.int64, count=len(rawids)),
        )
        return ak.unflatten(flat, ak.num(d["rawid"]))

    def _summary(self, period, run, string=None):
        """Binned counts of one run at base resolution.

        Shared across sessions: the entries are KB-scale but each is the
        reduction of a whole run's evt columns, so building one is the
        expensive part of a first visit to the Validation tab.
        """
        files = tuple(str(f) for f in self.viewer._run_files(period, run))

        def build():
            d = self._columns(period, run)
            t = d["timestamp"]
            if t.size == 0:
                return None
            hit_sel = (
                None if string is None else self._hit_selection(d, period, run, string)
            )
            # day-aligned absolute axis: every BIN_WIDTHS factor divides it
            # exactly, and bins of different runs line up with each other
            t0 = math.floor(t.min() / _DAY) * _DAY
            t1 = math.ceil(t.max() / _DAY) * _DAY
            t1 = max(t1, t0 + _DAY)
            axis = bh.axis.Regular(int((t1 - t0) / BASE_BIN_SECONDS), t0, t1)
            series = {}
            for skey, mask in self._series_masks(d, hit_sel).items():
                if mask is None:
                    series[skey] = None
                    continue
                h = bh.Histogram(axis)
                h.fill(t[mask])
                series[skey] = h
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
            return {"series": series, "exposure": exposure}

        key = (self.viewer.cycle_key, period, run, files, string)
        return VALIDATION_SUMMARIES.get(key, build)

    # -- period assembly ------------------------------------------------------

    def period_series(self, period, bin_seconds, string=None):
        """Mass-normalised rates of every series over all runs of ``period``.

        Returns ``(times_ms, rates)`` where ``times_ms`` are bin centres in
        ms-epoch (Bokeh datetime axis) and ``rates`` maps ``(group, label)`` to
        an array aligned with ``times_ms`` (``None`` when the underlying field
        is absent in every run). Runs are separated by a NaN row so lines
        break across gaps. ``string`` restricts events to hits in that string;
        rates are divided by the scope's ged mass (per run, from its
        channelmap). Rebinned from the cached base-resolution summaries; no
        file is re-read when ``bin_seconds`` changes.
        """
        if bin_seconds < BASE_BIN_SECONDS or bin_seconds % BASE_BIN_SECONDS:
            msg = (
                f"bin_seconds must be a positive multiple of {BASE_BIN_SECONDS}, "
                f"got {bin_seconds}"
            )
            raise ValueError(msg)
        factor = bin_seconds // BASE_BIN_SECONDS
        all_keys = [(g, lbl) for g, labels in RATE_GROUPS.items() for lbl in labels]
        times: list[np.ndarray] = []
        chunks: dict = {k: [] for k in all_keys}

        for run in sorted(self.viewer.available_runs().get(period, {})):
            summary = self._summary(period, run, string)
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
                h = summary["series"][k]
                if h is None:
                    chunks[k].append(np.full(exp_v.size, np.nan))
                    continue
                counts = h[:: bh.rebin(factor)].view()[sl].astype(float)
                unit = GROUP_UNIT_SECONDS[k[0]]
                chunks[k].append(counts / exp_v * unit / mass)

        if not times:
            return np.array([]), dict.fromkeys(all_keys)
        rates = {
            k: None if all(np.isnan(c).all() for c in parts) else np.concatenate(parts)
            for k, parts in chunks.items()
        }
        return np.concatenate(times), rates

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

        Loaded through dbetto's ``TextDB``, which merges the listed files the
        way the dataflow does. Shared across sessions: these files are
        MB-scale and take ~0.7 s to parse, and every session looking at this
        run wants the same ones.
        """
        tier, root, category, start_key, files = source

        def load():
            return TextDB(root, lazy=True).on(
                start_key, pattern=rf".*par_{tier}\.yaml$", category=category
            )

        # keyed by the resolved files, not the start key: every run the same
        # validity entry covers shares one parse
        pars = CAL_PARS.get((str(root), tuple(files)), load)
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
