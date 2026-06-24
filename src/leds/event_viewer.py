from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import lh5
import matplotlib as mpl
import numpy as np
from legendmeta import LegendMetadata
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from leds.config import load_paths


def _tstamp_to_unix(tstamp):
    """LEGEND file timestamp ``YYYYmmddTHHMMSSZ`` -> unix seconds."""
    return datetime.strptime(tstamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()


def parse_timestamp(text):
    """Parse user input into a unix timestamp.

    Accepts a raw unix value, the LEGEND ``YYYYmmddTHHMMSSZ`` form, or an ISO
    ``YYYY-mm-dd[ T]HH:MM:SS`` datetime (interpreted as UTC).
    """
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp()
        except ValueError:
            continue
    msg = f"could not parse timestamp: {text!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Detector geometry helpers (pure functions, framework-agnostic)
# ---------------------------------------------------------------------------


def is_coax(d):
    return d["type"] == "coax"


def is_taper(f):
    return f != {"angle_in_deg": 0, "height_in_mm": 0} and f != {
        "radius_in_mm": 0,
        "height_in_mm": 0,
    }


def is_bulletized(f):
    return "bulletization" in f and f["bulletization"] != {
        "top_radius_in_mm": 0,
        "bottom_radius_in_mm": 0,
        "borehole_radius_in_mm": 0,
        "contact_radius_in_mm": 0,
    }


def has_groove(f):
    return "groove" in f and f["groove"] != {
        "outer_radius_in_mm": 0,
        "depth_in_mm": 0,
        "width_in_mm": 0,
    }


def has_borehole(f):
    return "borehole" in f and f["borehole"] != {"gap_in_mm": 0, "radius_in_mm": 0}


def plot_geometry(d, R, H):
    coax = is_coax(d)

    g = d["geometry"]

    DH = g["height_in_mm"]
    DR = g["radius_in_mm"]

    xbot = []
    ybot = []

    botout = g["taper"]["bottom"]
    if is_taper(botout):
        TH = botout["height_in_mm"]
        TR = (
            botout["radius_in_mm"]
            if "radius_in_mm" in botout
            else TH * math.sin(botout["angle_in_deg"] * math.pi / 180)
        )
        xbot.extend([DR, DR - TR])
        ybot.extend([H - DH + TH, H - DH])
    else:
        xbot.append(DR)
        ybot.append(H - DH)

    if has_groove(g):
        GR = g["groove"]["radius_in_mm"]["outer"]
        GH = g["groove"]["depth_in_mm"]
        GW = g["groove"]["radius_in_mm"]["outer"] - g["groove"]["radius_in_mm"]["inner"]
        xbot.extend([GR, GR, GR - GW, GR - GW])
        ybot.extend([H - DH, H - DH + GH, H - DH + GH, H - DH])

    if coax:
        BG = g["borehole"]["depth_in_mm"]
        BR = g["borehole"]["radius_in_mm"]
        xbot.extend([BR, BR])
        ybot.extend([H - DH, H - DH + BG])

    xtop = []
    ytop = []

    topout = g["taper"]["top"]
    if is_taper(topout):
        TH = topout["height_in_mm"]
        TR = TH * math.sin(topout["angle_in_deg"] * math.pi / 180)
        xtop.extend([DR, DR - TR])
        ytop.extend([H - TH, H])
    else:
        xtop.append(DR)
        ytop.append(H)

    if has_borehole(g) and not coax:
        BG = g["borehole"]["depth_in_mm"]
        BR = g["borehole"]["radius_in_mm"]

        topin = g["taper"]["top"]
        if is_taper(topin):
            TH = topin["height_in_mm"]
            TR = TH * math.sin(topin["angle_in_deg"] * math.pi / 180)
            xtop.extend([BR + TR, BR, BR])
            ytop.extend([H, H - TH, H - DH + BG])
        else:
            xtop.extend([BR, BR])
            ytop.extend([H, H - DH + BG])

    x = np.hstack(
        (
            [-x + R for x in xbot],
            [x + R for x in xbot[::-1]],
            [x + R for x in xtop],
            [-x + R for x in xtop[::-1]],
        )
    )
    y = np.hstack((ybot, ybot[::-1], ytop, ytop[::-1]))
    return x, y


# per-system groupings: system -> {label -> (primary key, secondary sort, fmt)}
SYSTEM_GROUPINGS = {
    "geds": {
        "String": ("location.string", "location.position", "String:{:02}"),
        "HV filter": ("voltage.filter.id", "voltage.filter.channel", "HV:{}"),
        "CC4": ("electronics.cc4.id", "electronics.cc4.channel", "CC4:{}"),
    },
    "spms": {
        "barrel": ("location.barrel", "daq.rawid", "{}"),
    },
}
# kept for the geds-only array view
GED_GROUPINGS = SYSTEM_GROUPINGS["geds"]


def group_channels(chmap, system, grouping):
    """Map ``"<group label>" -> [rawid, ...]`` for one detector system.

    ``grouping`` selects how that system is partitioned (see ``SYSTEM_GROUPINGS``).
    """
    primary, secondary, fmt = SYSTEM_GROUPINGS[system][grouping]
    sub = chmap.map("system", unique=False)[system]
    by_primary = sub.map(primary, unique=False)
    out = {}
    for key in sorted(by_primary):
        members = by_primary[key].map(secondary)
        out[fmt.format(key)] = [members[s].daq.rawid for s in sorted(members)]
    return out


def group_geds(chmap, grouping="String"):
    """Map ``"<group label>" -> [rawid, ...]`` for the geds."""
    return group_channels(chmap, "geds", grouping)


def build_strings_dict(chmap):
    """Map ``"String:NN" -> [rawid, ...]`` (ordered by position) for the geds."""
    return group_geds(chmap, "String")


def get_plot_source(channel_map, strings_dict, dR=160, dH=40):
    """Build detector polygons laid out string-by-string.

    Returns ``(xs, ys, rawids)`` where ``rawids`` is the draw order, so a value
    array can be assembled in exactly the same order as the patches.
    """
    xs, ys, rawids = [], [], []
    R = 0
    for _name, string in strings_dict.items():
        H = 0
        for rawid in string:
            det = channel_map[rawid]
            x, y = plot_geometry(det, R, H)
            xs.append(x)
            ys.append(y)
            rawids.append(rawid)
            H -= det["geometry"]["height_in_mm"] + dH
        R += dR
    return xs, ys, rawids


# ---------------------------------------------------------------------------
# Event data + array view
# ---------------------------------------------------------------------------


class EventViewer:
    """Loads per-detector values for one event and renders the array view.

    Framework-agnostic: it owns a bare matplotlib ``Figure`` (Agg) that a Panel
    pane (or any host) can display. ``base_path`` selects the production cycle.

    Events are read from the ``evt`` tier, whose ``evt/geds`` group holds, per
    event, the variable-length list of fired detectors (``detector_name`` /
    ``rawid``) and their energies (``energy_field``). This is a single
    vector-of-vectors row read per event instead of one read per channel, so
    load times do not scale with the number of detectors.
    """

    def __init__(
        self,
        base_path=None,
        *,
        energy_field="energy",
        experiment="l200",
        datatype="phy",
    ):
        self.paths = load_paths(base_path)
        self.experiment = experiment
        self.datatype = datatype
        self.energy_field = energy_field

        # prefer the partition tiers (pet/pht/psp), fall back to evt/hit/dsp;
        # only the event tier is required, hit/dsp are optional (None if absent)
        self.tier = self._resolve_tier("pet", "evt")
        self.hit_tier = self._resolve_tier("pht", "hit", required=False)
        self.dsp_tier = self._resolve_tier("psp", "dsp", required=False)
        self.geds_group = f"{self.tier}/geds"

        self.meta = LegendMetadata(str(self.paths.metadata), lazy=True)
        self.status_db = LegendMetadata(str(self.paths.detector_status), lazy=True)

        self._n_events_cache: dict[str, int] = {}
        self._status_cache: dict = {}

        # populated by get_event()
        self.chmap = None
        self.tstamp = None
        self.event_timestamp = None
        self.period = None
        self.run = None
        self.index = None
        self.current_file = None
        self.multiplicity = None
        self.energy_dict: dict = {}
        self.fired_detectors: list[dict] = []
        self.spms_energy: dict = {}  # rawid -> summed trig-coincident energy
        self._status = None

    # -- filesystem discovery -------------------------------------------------

    def _resolve_tier(self, preferred, fallback, *, required=True):
        """Pick ``preferred`` tier if present, else ``fallback``.

        Returns ``None`` if neither exists and ``required`` is false; otherwise
        raises (used for the mandatory event tier).
        """
        for name in (preferred, fallback):
            key = f"tier_{name}"
            if key in self.paths and (Path(self.paths[key]) / self.datatype).is_dir():
                return name
        if required:
            msg = f"no {fallback} tier ({preferred}/{fallback}) found in this cycle"
            raise FileNotFoundError(msg)
        return None

    def _tier_root(self, tier):
        return Path(self.paths[f"tier_{tier}"]) / self.datatype

    def _run_files(self, period, run):
        root = self._tier_root(self.tier) / period / run
        pattern = (
            f"{self.experiment}-{period}-{run}-{self.datatype}-*-tier_{self.tier}.lh5"
        )
        return sorted(root.glob(pattern))

    def available_runs(self):
        """Scan the evt tier and return ``{period: {run: [timestamps]}}``."""
        tree: dict[str, dict[str, list[str]]] = {}
        root = self._tier_root(self.tier)
        if not root.is_dir():
            return tree
        for pdir in sorted(p for p in root.glob("p*") if p.is_dir()):
            for rdir in sorted(r for r in pdir.glob("r*") if r.is_dir()):
                tstamps = [
                    f.name.split("-")[4] for f in self._run_files(pdir.name, rdir.name)
                ]
                if tstamps:
                    tree.setdefault(pdir.name, {})[rdir.name] = tstamps
        return tree

    # -- event location -------------------------------------------------------

    def _n_events(self, file):
        file = str(file)
        if file not in self._n_events_cache:
            self._n_events_cache[file] = lh5.read_n_rows(
                f"{self.geds_group}/multiplicity", file
            )
        return self._n_events_cache[file]

    def run_length(self, period, run):
        """Total number of events across all files of a run."""
        return sum(self._n_events(f) for f in self._run_files(period, run))

    def _evt_file(self, period, run, tstamp):
        root = self._tier_root(self.tier) / period / run
        return root / (
            f"{self.experiment}-{period}-{run}-{self.datatype}"
            f"-{tstamp}-tier_{self.tier}.lh5"
        )

    def _periods(self):
        root = self._tier_root(self.tier)
        return sorted(p.name for p in root.glob("p*") if p.is_dir())

    def _runs(self, period):
        root = self._tier_root(self.tier) / period
        return sorted(r.name for r in root.glob("r*") if r.is_dir())

    def _run_tstamps(self, period, run):
        return [f.name.split("-")[4] for f in self._run_files(period, run)]

    def _run_start(self, period, run):
        tstamps = self._run_tstamps(period, run)
        return _tstamp_to_unix(tstamps[0]) if tstamps else None

    def _period_start(self, period):
        for run in self._runs(period):
            start = self._run_start(period, run)
            if start is not None:
                return start
        return None

    @staticmethod
    def _narrow(candidates, start_of, target):
        """Last candidate whose start <= target (else the earliest)."""
        chosen, chosen_start = candidates[0], None
        for c in candidates:
            start = start_of(c)
            if (
                start is not None
                and start <= target
                and (chosen_start is None or start >= chosen_start)
            ):
                chosen, chosen_start = c, start
        return chosen

    def locate_timestamp(self, target):
        """Find the event nearest ``target`` (unix s); return ``(period, run, idx)``.

        Narrows hierarchically using start timestamps — period, then run, then
        file (directory listings only) — and reads just the one matching file's
        ``evt/trigger/timestamp`` column to binary-search the event.
        """
        periods = self._periods()
        if not periods:
            msg = "no evt files to search"
            raise FileNotFoundError(msg)
        period = self._narrow(periods, self._period_start, target)
        run = self._narrow(
            self._runs(period), lambda r: self._run_start(period, r), target
        )
        tstamps = self._run_tstamps(period, run)
        tstamp = self._narrow(tstamps, _tstamp_to_unix, target)
        file = self._evt_file(period, run, tstamp)

        times = lh5.read(f"{self.tier}/trigger/timestamp", str(file)).nda
        local = int(np.searchsorted(times, target))
        if local >= len(times):
            local = len(times) - 1
        elif local > 0 and abs(times[local - 1] - target) <= abs(times[local] - target):
            local -= 1

        preceding = 0
        for run_file in self._run_files(period, run):
            if run_file == file:
                break
            preceding += self._n_events(run_file)
        return period, run, preceding + local

    def locate(self, period, run, idx):
        """Map a run-global ``idx`` to ``(file, local_row)`` across the run files."""
        files = self._run_files(period, run)
        if not files:
            msg = f"no {self.tier} files for {period}/{run} under {self._tier_root(self.tier)}"
            raise FileNotFoundError(msg)
        counts = []
        for file in files:
            counts.append(self._n_events(file))
            if sum(counts) > idx:
                break
        cum = np.cumsum(counts)
        if idx >= cum[-1]:
            msg = f"event index {idx} out of range (run has {int(cum[-1])} events)"
            raise IndexError(msg)
        file_n = int(np.argmin(np.where(cum - idx > 0, cum, np.inf)))
        local = idx - (cum[file_n - 1] if file_n > 0 else 0)
        return files[file_n], int(local)

    # -- event data -----------------------------------------------------------

    def _read_row(self, file, field, local, group=None):
        """Read one variable-length ``evt`` group row as a Python list."""
        group = group or self.geds_group
        vov = lh5.read(f"{group}/{field}", str(file), start_row=local, n_rows=1)
        return vov.view_as("ak").to_list()[0]

    def _status_for(self, tstamp):
        if tstamp not in self._status_cache:
            self._status_cache[tstamp] = self.status_db.statuses.on(tstamp)
        return self._status_cache[tstamp]

    def usability(self, name):
        """Detector usability ("on"/"ac"/"off") from the status DB for this event."""
        if self._status is not None and name in self._status:
            return str(self._status[name].usability)
        return "off"

    def get_event(self, period, run, idx):
        file, local = self.locate(period, run, idx)
        self.current_file = file
        self.period = period
        self.run = run
        self.index = local
        self.tstamp = file.name.split("-")[4]
        self.event_timestamp = float(
            lh5.read(
                f"{self.tier}/trigger/timestamp", str(file), start_row=local, n_rows=1
            ).nda[0]
        )
        self.chmap = self.meta.channelmap(self.tstamp)

        # Seed every ged so the array view distinguishes "working but no hit"
        # (0 -> grey, under threshold) from "not processable" (None -> white).
        geds = self.chmap.map("system", unique=False).geds
        energies: dict = {
            det: (0 if self.chmap[det].analysis.processable else None)
            for det in geds.map("name")
        }

        # Single evt/geds row read gives the fired detectors, their energies and
        # the index back into each channel's raw/hit table (for waveforms).
        names = [
            n.decode() if isinstance(n, (bytes, bytearray)) else n
            for n in self._read_row(file, "detector_name", local)
        ]
        values = self._read_row(file, self.energy_field, local)
        rawids = self._read_row(file, "rawid", local)
        hit_idxs = self._read_row(file, "hit_idx", local)

        self.fired_detectors = []
        for name, val, rawid, hit_idx in zip(
            names, values, rawids, hit_idxs, strict=True
        ):
            energy = 0.0 if val is None or np.isnan(val) else float(val)
            energies[name] = energy
            self.fired_detectors.append(
                {
                    "name": name,
                    "rawid": int(rawid),
                    "hit_idx": int(hit_idx),
                    "energy": energy,
                }
            )

        self.energy_dict = energies
        self.multiplicity = len(names)

        # SiPMs: per channel, sum the energy of its triggered coincident pulses
        # (evt/spms/energy and is_trig_coin_pulse are nested per pulse).
        self._status = self._status_for(self.tstamp)
        spm_rawids = self._read_row(file, "rawid", local, group=f"{self.tier}/spms")
        spm_trig = self._read_row(
            file, "is_trig_coin_pulse", local, group=f"{self.tier}/spms"
        )
        spm_energy = self._read_row(file, "energy", local, group=f"{self.tier}/spms")
        self.spms_energy = {}
        for rawid, pulses, pulse_energy in zip(
            spm_rawids, spm_trig, spm_energy, strict=True
        ):
            # sum of spms.energy over the channel's triggered coincident pulses
            self.spms_energy[int(rawid)] = sum(
                e for trig, e in zip(pulses, pulse_energy, strict=True) if trig
            )
        return self

    # -- waveforms ------------------------------------------------------------

    def _raw_root(self):
        """Raw tier root, falling back to ``<tier>/raw`` when the configured
        ``tier_raw`` path is absent (e.g. blinded/relocated raw)."""
        configured = Path(self.paths["tier_raw"])
        if configured.is_dir():
            return configured
        return Path(self.paths["tier"]) / "raw"

    def raw_file(self):
        """Path to the raw-tier file matching the current event's evt file."""
        name = self.current_file.name.replace(f"tier_{self.tier}", "tier_raw")
        return self._raw_root() / self.datatype / self.period / self.run / name

    def read_waveform(self, rawid, hit_idx, kind="waveform_windowed"):
        """Return ``(times_ns, values)`` for one detector's raw waveform."""
        wf = lh5.read(
            f"ch{rawid:07d}/raw/{kind}",
            str(self.raw_file()),
            start_row=hit_idx,
            n_rows=1,
        )
        values = wf.values.nda[0]
        t0 = float(wf.t0.nda[0])
        dt = float(wf.dt.nda[0])
        times = t0 + dt * np.arange(len(values))
        return times, values

    # -- rendering ------------------------------------------------------------

    def plot(self, fig=None, *, vmin=25, vmax=6000, figsize=(7, 10)):
        if self.chmap is None:
            msg = "call get_event() before plot()"
            raise RuntimeError(msg)

        fig = fig if fig is not None else Figure(figsize=figsize)
        fig.clf()
        ax = fig.add_subplot(111)

        channel_map = self.chmap.map("daq.rawid")
        strings_dict = build_strings_dict(self.chmap)
        xs, ys, rawids = get_plot_source(channel_map, strings_dict)

        patches = [
            Polygon(np.column_stack([x, y]), closed=True)
            for x, y in zip(xs, ys, strict=True)
        ]
        values = np.array(
            [
                v
                if (v := self.energy_dict.get(channel_map[r]["name"])) is not None
                else np.nan
                for r in rawids
            ],
            dtype=float,
        )

        cmap = mpl.colormaps["viridis"].with_extremes(under="grey", bad="white")
        coll = PatchCollection(
            patches, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax), edgecolor="black"
        )
        coll.set_array(values)
        ax.add_collection(coll, autolim=True)
        ax.autoscale_view()
        ax.margins(0.05)

        fig.colorbar(coll, ax=ax, label=f"{self.energy_field} (keV)")
        mult = "" if self.multiplicity is None else f" M={self.multiplicity}"
        ax.set_title(f"{self.experiment}-{self.tstamp} idx:{self.index}{mult}")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig
