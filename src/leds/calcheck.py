"""Calibration check: every detector's whole cal-run spectrum, every cal run.

A few files of each cal run (``FILES_PER_RUN``, spread over the run) hold
~50 k hits per detector, plenty for the spectrum shape. They are reduced to
per-detector 0.5 keV histograms, cached and shared across sessions, and built
in the background, one run at a time:

- *uncut*, from the evt/pet tier (energy + rawid), ~1.5 s a run;
- *section data* for ``is_valid_cal``, from 2 files of the hit/pht tier,
  15-30 s a run cold:
  per detector a histogram of the hits passing every section, plus the
  energies and section bitmask of the few that fail one. Any AND of sections
  is then exact without re-reading.

:func:`scale_match` compares each spectrum with the period's median spectrum
in log-energy, where a gain error is a shift. It gives the energy-scale error
in %, and how well the shape matches: a wrong peak identification shows as a
large error or a poor match, whichever peak lands at 2614 keV.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import awkward as ak
import h5py
import lh5
import numexpr as ne
import numpy as np

from leds._cache import CAL_CHECK_CUTS, CAL_CHECK_SPECTRA
from leds.event_viewer import EventViewer
from leds.validation import ENERGY_PARAM, _read_groups, load_par_files

FILES_PER_RUN = 3
# the hit tier is ~10x slower to read (all channels x ~14 fields): 2 files
# (~35 k hits per detector) keep a period's section data to minutes
CUT_FILES_PER_RUN = 2
BIN = 0.5  # keV, stored spectra
E_MAX = 3000.0
N_BINS = int(E_MAX / BIN)
DISPLAY_BIN = 2.0  # keV, drill-down plots
TH_LINES = (238.6, 583.2, 727.3, 860.6, 1592.5, 2103.5, 2614.5)
MIN_COUNTS = 5000  # hits below which a spectrum is not matched
MAX_SCALE = 1.35  # largest energy-scale factor searched, either way
LOG_E = np.linspace(np.log(200.0), np.log(2750.0), 6000)
SMOOTH = 120  # log-energy points (~2%) in the continuum running mean
CAL_SECTIONS_OF = "is_valid_cal"

_BUILD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="leds-calcheck")
_pending: set = set()
_pending_lock = threading.Lock()


# -- pure functions --------------------------------------------------------------


def pick_files(files, n=FILES_PER_RUN):
    """``n`` files spread evenly over a run's sorted file list."""
    files = sorted(files)
    if len(files) <= n:
        return files
    return [files[i] for i in np.linspace(0, len(files) - 1, n).astype(int)]


def hist_by_rawid(energy, rawid, rawids):
    """``{rawid: 0.5 keV histogram}`` of ``energy`` for each of ``rawids``."""
    rawids = np.asarray(sorted(rawids), dtype=np.int64)
    if rawids.size == 0:
        return {}
    e = np.asarray(energy, dtype=float)
    r = np.asarray(rawid, dtype=np.int64)
    pos = np.minimum(np.searchsorted(rawids, r), rawids.size - 1)
    ok = (rawids[pos] == r) & np.isfinite(e) & (e >= 0) & (e < E_MAX)
    key = pos[ok] * N_BINS + (e[ok] / BIN).astype(np.int64)
    counts = np.bincount(key, minlength=rawids.size * N_BINS).reshape(-1, N_BINS)
    return {int(rid): counts[i].astype(np.float32) for i, rid in enumerate(rawids)}


def _identifiers(expression):
    """Variable names in an expression (names called as functions excluded)."""
    return {
        m.group()
        for m in re.finditer(r"[A-Za-z_]\w*", expression)
        if not expression[m.end() :].lstrip().startswith("(")
    }


def section_names(ops):
    """The ``&``-terms of the ``is_valid_cal`` expression, in order."""
    expr = " ".join(ops[CAL_SECTIONS_OF]["expression"].split())
    terms = [t.strip().strip("()").strip() for t in expr.split("&")]
    if not all(re.fullmatch(r"[A-Za-z_]\w*", t) for t in terms):
        return [CAL_SECTIONS_OF]  # not a plain AND: offer it whole
    return terms


def needed_fields(name, ops, stored, _depth=0):
    """Stored hit fields that ``name`` is computed from (``KeyError`` if not)."""
    if name in stored:
        return {name}
    if _depth > 20:
        raise KeyError(name)
    op = ops[name]
    params = set(op.get("parameters") or {})
    out: set = set()
    for ident in _identifiers(op["expression"]) - params:
        out |= needed_fields(ident, ops, stored, _depth + 1)
    return out


def evaluate(name, ops, cols, _depth=0):
    """``name`` from stored columns ``cols``, via the pars' expressions."""
    if name in cols:
        return cols[name]
    if _depth > 20:
        raise KeyError(name)
    op = ops[name]
    params = dict(op.get("parameters") or {})
    local = {
        i: evaluate(i, ops, cols, _depth + 1)
        for i in _identifiers(op["expression"]) - set(params)
    }
    return ne.evaluate(" ".join(op["expression"].split()), local_dict=local | params)


def cut_hist(det, selected):
    """``det``'s spectrum with the ``selected`` sections applied (AND).

    ``det`` is one detector's section data (see :meth:`CalCheck._build_cuts`).
    Selected sections it lacks are not applied; they come back as the second
    element.
    """
    names = det["sections"]
    bits = 0
    for s in selected:
        if s in names:
            bits |= 1 << names.index(s)
    missing = [s for s in selected if s not in names or s in det["missing"]]
    keep = (det["fail_mask"] & bits) == bits
    extra, _ = np.histogram(det["fail_e"][keep], bins=N_BINS, range=(0, E_MAX))
    return det["pass"] + extra.astype(np.float32), missing


def profiles(spectra):
    """Peaks-only log-spectra on the ``LOG_E`` grid: zero mean, unit norm."""
    centers = (np.arange(N_BINS) + 0.5) * BIN
    grid = np.exp(LOG_E)
    y = np.log1p(np.array([np.interp(grid, centers, s) for s in spectra]))
    csum = np.cumsum(
        np.pad(y, ((0, 0), (SMOOTH // 2, SMOOTH - SMOOTH // 2)), mode="edge"), axis=1
    )
    base = (csum[:, SMOOTH:] - csum[:, :-SMOOTH]) / SMOOTH
    p = np.clip(y - base[:, : y.shape[1]], 0, None)
    p -= p.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    return np.divide(p, norm, out=np.zeros_like(p), where=norm > 0)


def scale_match(spectra, reference=None):
    """Energy-scale error (%) and shape match of each spectrum.

    ``spectra`` is ``(n, N_BINS)``. Each is cross-correlated in log-energy
    with ``reference`` (default: the median profile of those with at least
    ``MIN_COUNTS`` hits), over scale factors up to ``MAX_SCALE`` either way.
    Returns ``(error_pct, match)``, NaN below ``MIN_COUNTS``. The error is
    how much too high the energy scale is.
    """
    spectra = np.asarray(spectra, dtype=float)
    n = spectra.shape[0]
    err = np.full(n, np.nan)
    match = np.full(n, np.nan)
    enough = spectra.sum(axis=1) >= MIN_COUNTS
    if not enough.any():
        return err, match
    prof = profiles(spectra[enough])
    ref = np.median(prof, axis=0) if reference is None else reference
    ref = ref / (np.linalg.norm(ref) or 1.0)
    size = 2 * prof.shape[1]
    cc = np.fft.irfft(np.fft.rfft(prof, size) * np.conj(np.fft.rfft(ref, size)), size)
    du = LOG_E[1] - LOG_E[0]
    lag = int(np.log(MAX_SCALE) / du)
    window = np.concatenate([cc[:, size - lag :], cc[:, : lag + 1]], axis=1)
    k = np.argmax(window, axis=1)
    rows = np.arange(window.shape[0])
    # parabola through the peak for a sub-point lag
    inner = (k > 0) & (k < window.shape[1] - 1)
    a = window[rows, np.maximum(k - 1, 0)]
    b = window[rows, k]
    c = window[rows, np.minimum(k + 1, window.shape[1] - 1)]
    denom = a - 2 * b + c
    sub = np.where(
        inner & (denom != 0), 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0
    )
    err[enough] = (np.exp((k - lag + sub) * du) - 1) * 100
    match[enough] = b
    return err, match


def coarsen(spectrum, width=DISPLAY_BIN):
    """Sum 0.5 keV bins into ``width`` keV bins."""
    k = round(width / BIN)
    return np.asarray(spectrum)[: N_BINS // k * k].reshape(-1, k).sum(axis=1)


# -- data --------------------------------------------------------------------------


def _submit(cache, key, factory):
    """Build ``cache[key]`` in the background, once; failures are cached too."""
    with _pending_lock:
        if key in _pending or cache.peek(key) is not None:
            return
        _pending.add(key)

    def job():
        try:
            cache.get(key, lambda: _safely(factory))
        finally:
            with _pending_lock:
                _pending.discard(key)

    _BUILD.submit(job)


def _safely(factory):
    try:
        return factory()
    except Exception as exc:  # shown in the plot as unavailable
        return {"error": f"{type(exc).__name__}: {exc}"}


class CalCheck:
    """Calibration-check data for one production cycle's cal runs."""

    def __init__(self, cycle_dir):
        self.cycle_dir = Path(cycle_dir)
        self._viewer = None
        self._lock = threading.Lock()

    @property
    def viewer(self):
        with self._lock:
            if self._viewer is None:
                self._viewer = EventViewer(self.cycle_dir, datatype="cal")
            return self._viewer

    def runs(self, period):
        return sorted(self.viewer.available_runs().get(period, {}))

    def detectors(self, period):
        """``[(label, name, rawid, string)]`` of the geds, in string order."""
        runs = self.runs(period)
        if not runs:
            return []
        tstamp = self.viewer._run_tstamps(period, runs[0])[0]
        geds = self.viewer._channelmap(tstamp).map("system", unique=False).geds
        dets = sorted(
            geds.map("name").values(),
            key=lambda d: (int(d.location.string), int(d.location.position)),
        )
        return [
            (
                f"s{int(d.location.string):02d} {d.name}",
                d.name,
                int(d.daq.rawid),
                int(d.location.string),
            )
            for d in dets
        ]

    # keys: the files read, so a run that gains a file is a new entry
    def _evt_files(self, period, run):
        return tuple(str(f) for f in pick_files(self.viewer._run_files(period, run)))

    def _hit_files(self, period, run):
        tier = self.viewer.hit_tier
        if tier is None:
            return ()
        root = self.viewer._tier_root(tier) / period / run
        return tuple(str(f) for f in pick_files(root.glob("*.lh5"), CUT_FILES_PER_RUN))

    def _build_spectra(self, period, run, rawids):
        files = self._evt_files(period, run)
        raw = _read_groups(self.viewer.group, files, {"geds": ("energy", "rawid")})
        if raw["geds/energy"] is None or raw["geds/rawid"] is None:
            msg = f"no geds/energy in the {self.viewer.tier} cal tier"
            raise KeyError(msg)
        e = ak.to_numpy(ak.flatten(raw["geds/energy"].view_as("ak")))
        r = ak.to_numpy(ak.flatten(raw["geds/rawid"].view_as("ak")))
        return hist_by_rawid(e, r, rawids)

    def _cal_pars(self, period, run):
        tier = self.viewer.hit_tier
        root = Path(self.viewer.paths[f"par_{tier}"])
        files = sorted(
            str(p.relative_to(root))
            for p in (root / "cal" / period / run).glob(f"*-par_{tier}.yaml")
        )
        if not files:
            msg = f"no par_{tier} file for cal run {period} {run}"
            raise FileNotFoundError(msg)
        return load_par_files(root, files)

    def _build_cuts(self, period, run, dets):
        """Per rawid: passing histogram, failing energies and section masks."""
        files = self._hit_files(period, run)
        if not files:
            msg = f"no {self.viewer.hit_tier} files for cal run {period} {run}"
            raise FileNotFoundError(msg)
        pars = self._cal_pars(period, run)
        out = {}
        handles = []
        try:
            for f in files:
                handles.append(h5py.File(f, "r", locking=False))
            channels = set(lh5.ls(handles[0]))
            for _label, name, rid, _string in dets:
                ch = f"ch{rid}"
                if ch not in channels or name not in pars:
                    continue
                ops = pars[name]["pars"]["operations"]
                stored = {x.split("/")[-1] for x in lh5.ls(handles[0], f"{ch}/hit/")}
                if ENERGY_PARAM not in stored or CAL_SECTIONS_OF not in ops:
                    continue
                sections = section_names(ops)
                fields, missing = {ENERGY_PARAM}, set()
                for s in sections:
                    try:
                        fields |= needed_fields(s, ops, stored)
                    except KeyError:
                        missing.add(s)
                tbl = lh5.read(f"{ch}/hit", handles, field_mask=sorted(fields))
                cols = {k: tbl[k].nda for k in tbl}
                e = cols[ENERGY_PARAM].astype(float)
                mask = np.zeros(e.size, dtype=np.uint8)
                for k, s in enumerate(sections):
                    ok = np.ones(e.size, dtype=bool)
                    if s not in missing:
                        ok = np.asarray(evaluate(s, ops, cols), dtype=bool)
                    mask |= ok.astype(np.uint8) << k
                full = (1 << len(sections)) - 1
                good = np.isfinite(e)
                passing = good & (mask == full)
                h, _ = np.histogram(e[passing], bins=N_BINS, range=(0, E_MAX))
                failing = good & (mask != full)
                out[rid] = {
                    "sections": sections,
                    "missing": missing,
                    "pass": h.astype(np.float32),
                    "fail_e": e[failing],  # full precision: bins stay exact
                    "fail_mask": mask[failing],
                }
        finally:
            for h in handles:
                h.close()
        return out

    # -- queries: never block on a build ----------------------------------------------

    def ensure(self, period, *, cuts=True):
        """Queue the period's missing builds: uncut first, then section data."""
        dets = self.detectors(period)
        rawids = [d[2] for d in dets]
        runs = self.runs(period)
        for run in runs:
            _submit(
                CAL_CHECK_SPECTRA,
                self._evt_files(period, run),
                lambda run=run: self._build_spectra(period, run, rawids),
            )
        if cuts:
            for run in runs:
                _submit(
                    CAL_CHECK_CUTS,
                    ("cuts", *self._hit_files(period, run)),
                    lambda run=run: self._build_cuts(period, run, dets),
                )

    def progress(self, period):
        """``(uncut ready, section data ready, cal runs)``, errors counting as ready."""
        runs = self.runs(period)
        spectra = sum(
            CAL_CHECK_SPECTRA.peek(self._evt_files(period, r)) is not None for r in runs
        )
        cuts = sum(
            CAL_CHECK_CUTS.peek(("cuts", *self._hit_files(period, r))) is not None
            for r in runs
        )
        return spectra, cuts, len(runs)

    def section_labels(self, period):
        """Section names offered for the period (from the first built run)."""
        for run in self.runs(period):
            data = CAL_CHECK_CUTS.peek(("cuts", *self._hit_files(period, run)))
            if data and "error" not in data:
                for det in data.values():
                    return list(det["sections"])
        return []

    def spectra(self, period, selected=()):
        """What the plots draw, from whatever is built so far.

        Returns ``{"runs", "dets", "counts": (n_det, n_run, N_BINS), "ready":
        (n_det, n_run) bool, "errors": {run: reason}, "missing": {label:
        sections not applied}}``. With ``selected`` sections the section data
        is used, else the uncut spectra.
        """
        runs = self.runs(period)
        dets = self.detectors(period)
        counts = np.zeros((len(dets), len(runs), N_BINS), dtype=np.float32)
        ready = np.zeros((len(dets), len(runs)), dtype=bool)
        errors, missing = {}, {}
        for j, run in enumerate(runs):
            if selected:
                data = CAL_CHECK_CUTS.peek(("cuts", *self._hit_files(period, run)))
            else:
                data = CAL_CHECK_SPECTRA.peek(self._evt_files(period, run))
            if data is None:
                continue
            if "error" in data:
                errors[run] = data["error"]
                continue
            for i, (label, _name, rid, _string) in enumerate(dets):
                entry = data.get(rid)
                if entry is None:
                    continue
                if selected:
                    counts[i, j], lacking = cut_hist(entry, selected)
                    if lacking:
                        missing.setdefault(label, set()).update(lacking)
                else:
                    counts[i, j] = entry
                ready[i, j] = True
        return {
            "runs": runs,
            "dets": dets,
            "counts": counts,
            "ready": ready,
            "errors": errors,
            "missing": missing,
        }
