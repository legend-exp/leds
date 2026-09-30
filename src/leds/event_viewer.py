from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import h5py
import lh5
import numpy as np
from dbetto.catalog import Catalog
from legendmeta import LegendMetadata
from packaging.version import Version

from leds._cache import (
    CATALOGS,
    CHANNELMAPS,
    METADATA,
    N_EVENTS,
    RUN_FILES,
    RUN_TREES,
    STATUSES,
)
from leds.config import load_paths

VALIDITY_FILES = ("validity.yaml", "validity.jsonl")


def _valid_files(db, tstamp, category="all"):
    """The files ``db.on(tstamp, category=category)`` would merge, as a tuple.

    Serves as a cache key: every timestamp of one validity entry gets the same
    one. ``None`` when that cannot be worked out, so callers key by the
    timestamp instead -- slower, never wrong.
    """
    try:
        root = Path(db.__path__)
        found = [root / f for f in VALIDITY_FILES if (root / f).is_file()]
        if len(found) != 1:
            return None
        validity = str(found[0])
        catalog = CATALOGS.get((validity,), lambda: Catalog.read_from(validity))
        return tuple(str(f) for f in catalog.valid_for(tstamp, category))
    except Exception:  # unexpected layout: fall back to the timestamp
        return None


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

#: Ged seeds derived from a channelmap, kept per viewer. A period can span more
#: timestamps than this, so eviction must drop the *oldest* entry rather than
#: the whole cache -- the Dataset tab walks every run of a cycle in order, and
#: clearing would make it rebuild everything it had just built.
MAX_CACHED_CHANNELMAPS = 16

#: Static per-channelmap glyph geometry cached on behalf of the view modules.
MAX_CACHED_VIEWS = 64

#: Raw/evt files held open per session. Small: an event only ever touches its
#: own run's files, and each handle holds read buffers.
MAX_OPEN_FILES = 4

#: First metadata tag laying out the analysis map under ``datasets.statuses``;
#: mirrors the check inside ``LegendMetadata.channelmap``.
_MIN_STATUSES_TAG = Version("v0.5.9")


def _evict(cache, max_items):
    """Drop the least-recently-inserted entries until ``cache`` fits."""
    while len(cache) > max_items:
        cache.pop(next(iter(cache)))


def _shared_metadata(path):
    """One ``LegendMetadata`` per checkout, shared by every session.

    Building one resolves paths and opens a git repo, and its ``TextDB`` file
    store is what makes runinfo and groupings cheap after the first read --
    all of it identical for every session pointing at the same checkout.
    """
    path = str(path)
    return METADATA.get((path,), lambda: LegendMetadata(path, lazy=True))


class EventViewer:
    """Loads per-detector values for one event and renders the array view.

    Framework-agnostic: it only reads and reshapes data, leaving rendering to
    the view modules. ``base_path`` selects the production cycle.

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
        # in-file LH5 groups keep the base product name (evt/hit/dsp) even when
        # the files live in the partitioned tier (pet/pht/psp), so reads must be
        # prefixed with the group, not the file tier.
        self.group = "evt"
        self.hit_group = "hit" if self.hit_tier else None
        self.dsp_group = "dsp" if self.dsp_tier else None
        self.geds_group = f"{self.group}/geds"

        self.meta = _shared_metadata(self.paths.metadata)
        self.status_db = _shared_metadata(self.paths.detector_status)

        self._geds_seed_cache: dict = {}
        self._raw_root_cache = None
        self._open_files: dict = {}  # path -> h5py.File (per-session LRU)
        self._skip_version_check = None  # resolved once, in _channelmap
        #: scratch cache for view modules (static per-channelmap glyph data,
        #: keyed by ("<what>", tstamp, ...)); pruned in get_event()
        self.view_cache: dict = {}

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

    @property
    def cycle_key(self):
        """Stable identity of this cycle's event data, for shared cache keys.

        Two viewers with the same key read the same files, so an entry built
        by one session is valid for another.
        """
        return str(self._tier_root(self.tier))

    def _run_files(self, period, run):
        # shared and TTL'd: this is called per event (locate/run_length) and
        # each call otherwise re-globs a possibly slow network filesystem
        root = self._tier_root(self.tier)
        key = (str(root), period, run)

        def scan():
            pattern = (
                f"{self.experiment}-{period}-{run}-{self.datatype}"
                f"-*-tier_{self.tier}.lh5"
            )
            return sorted((root / period / run).glob(pattern))

        return RUN_FILES.get(key, scan)

    def available_runs(self):
        """Scan the evt tier and return ``{period: {run: [timestamps]}}``.

        Shared across sessions but short-lived: new runs land while the server
        is up, and users expect to see them without a restart.
        """
        root = self._tier_root(self.tier)

        def scan():
            tree: dict[str, dict[str, list[str]]] = {}
            if not root.is_dir():
                return tree
            for pdir in sorted(p for p in root.glob("p*") if p.is_dir()):
                for rdir in sorted(r for r in pdir.glob("r*") if r.is_dir()):
                    tstamps = self._run_tstamps(pdir.name, rdir.name)
                    if tstamps:
                        tree.setdefault(pdir.name, {})[rdir.name] = tstamps
            return tree

        return RUN_TREES.get((str(root),), scan)

    # -- event location -------------------------------------------------------

    def _n_events(self, file):
        # shared and never expired: a written evt file does not change
        file = str(file)
        return N_EVENTS.get(
            (file, self.geds_group),
            lambda: lh5.read_n_rows(f"{self.geds_group}/multiplicity", file),
        )

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

        times = lh5.read(f"{self.group}/trigger/timestamp", str(file)).nda
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

    def _read_table_row(self, file, group, local, fields):
        """Read one row of several ``evt`` table fields in a single file open.

        Returns ``{field: python value}`` (variable-length fields as lists).
        """
        tbl = lh5.read(
            group, str(file), start_row=local, n_rows=1, field_mask=list(fields)
        )
        return {f: tbl[f].view_as("ak").to_list()[0] for f in fields}

    def statuses(self, start_key, category=None):
        """Detector statuses at ``start_key``, shared across sessions.

        ``TextDB.on`` re-parses ``validity.yaml`` on *every* call, and the
        dataset views ask for one status set per run of the cycle -- so this
        is the accessor those should use rather than reaching into
        ``status_db.statuses`` themselves.
        """
        files = _valid_files(self.status_db.statuses, start_key, category or "all")
        key = (str(self.paths.detector_status), files or start_key, category)
        if category is None:
            return STATUSES.get(key, lambda: self.status_db.statuses.on(start_key))
        return STATUSES.get(
            key, lambda: self.status_db.statuses.on(start_key, category=category)
        )

    def _status_for(self, tstamp):
        return self.statuses(tstamp)

    def usability(self, name):
        """Detector usability ("on"/"ac"/"off") from the status DB for this event."""
        if self._status is not None and name in self._status:
            return str(self._status[name].usability)
        return "off"

    def _skip_git_version_check(self):
        """Whether ``channelmap()`` may assume the post-v0.5.9 metadata layout.

        Left to itself, ``channelmap()`` decides this by shelling out to
        ``git describe`` on *every* call -- the bulk of a warm channelmap
        build, and worse still on a network-mounted ``.git``. The answer only
        changes when the checkout does, so resolve it once per viewer.
        """
        if self._skip_version_check is None:
            try:
                self._skip_version_check = bool(
                    self.meta.__closest_tag__ >= _MIN_STATUSES_TAG
                    and self.meta.__version__ != str(_MIN_STATUSES_TAG)
                )
            except Exception:
                # not a git checkout (or no tags): channelmap() documents
                # skip_version_check for exactly this case
                self._skip_version_check = True
        return self._skip_version_check

    def _channelmap(self, tstamp):
        """Channelmap for ``tstamp``, shared across sessions.

        The single most expensive thing a session does: building one parses
        every detector YAML in the checkout. The result is a deep copy marked
        read-only all the way down, so sharing it cannot let one session
        corrupt another's view.

        Keyed by the files valid at ``tstamp`` in every db the build reads
        (``dataprod.config`` only matters for pre-v0.5.9 metadata), so all
        DAQ files of one validity entry share one build.
        """
        meta = self.meta
        try:
            dbs = (
                meta.hardware.configuration.channelmaps,
                meta.datasets.statuses,
                meta.dataprod.config,
            )
        except Exception:  # an unusual checkout: key by the timestamp
            dbs = ()
        files = [_valid_files(db, tstamp) for db in dbs]
        key = tuple(files) if files and None not in files else tstamp
        return CHANNELMAPS.get(
            (str(self.paths.metadata), key),
            lambda: self.meta.channelmap(
                tstamp, skip_version_check=self._skip_git_version_check()
            ),
        )

    def _geds_seed(self, tstamp):
        """Baseline ``{detector: 0 | None}`` energies for every ged (cached).

        Seeds every ged so the array view distinguishes "working but no hit"
        (0 -> grey, under threshold) from "not processable" (None -> white).
        """
        seed = self._geds_seed_cache.pop(tstamp, None)  # re-inserted below
        if seed is None:
            geds = self.chmap.map("system", unique=False).geds
            seed = {
                det: (0 if self.chmap[det].analysis.processable else None)
                for det in geds.map("name")
            }
        self._geds_seed_cache[tstamp] = seed
        _evict(self._geds_seed_cache, MAX_CACHED_CHANNELMAPS)
        return dict(seed)

    def get_event(self, period, run, idx):
        file, local = self.locate(period, run, idx)
        self.current_file = file
        self.period = period
        self.run = run
        self.index = local
        self.tstamp = file.name.split("-")[4]
        self.event_timestamp = float(
            lh5.read(
                f"{self.group}/trigger/timestamp", str(file), start_row=local, n_rows=1
            ).nda[0]
        )
        self.chmap = self._channelmap(self.tstamp)
        _evict(self.view_cache, MAX_CACHED_VIEWS)

        energies = self._geds_seed(self.tstamp)

        # Single evt/geds row read gives the fired detectors, their energies and
        # the index back into each channel's raw/hit table (for waveforms).
        geds_row = self._read_table_row(
            file,
            self.geds_group,
            local,
            ("detector_name", self.energy_field, "rawid", "hit_idx"),
        )
        names = [
            n.decode() if isinstance(n, bytes | bytearray) else n
            for n in geds_row["detector_name"]
        ]
        values = geds_row[self.energy_field]
        rawids = geds_row["rawid"]
        hit_idxs = geds_row["hit_idx"]

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
        spms_row = self._read_table_row(
            file,
            f"{self.group}/spms",
            local,
            ("rawid", "is_trig_coin_pulse", "energy"),
        )
        self.spms_energy = {}
        for rawid, pulses, pulse_energy in zip(
            spms_row["rawid"],
            spms_row["is_trig_coin_pulse"],
            spms_row["energy"],
            strict=True,
        ):
            # sum of spms.energy over the channel's triggered coincident pulses
            self.spms_energy[int(rawid)] = sum(
                e for trig, e in zip(pulses, pulse_energy, strict=True) if trig
            )
        return self

    # -- waveforms ------------------------------------------------------------

    def _raw_root(self):
        """Raw tier root, falling back to ``<tier>/raw`` when the configured
        ``tier_raw`` path is absent (e.g. blinded/relocated raw).

        Memoised: this is hit once per waveform trace, and the all-waveforms
        view draws ~100 of them per render -- that many ``is_dir`` stats is a
        real cost on a network filesystem.
        """
        if self._raw_root_cache is None:
            configured = Path(self.paths["tier_raw"])
            self._raw_root_cache = (
                configured if configured.is_dir() else Path(self.paths["tier"]) / "raw"
            )
        return self._raw_root_cache

    def raw_file(self):
        """Path to the raw-tier file matching the current event's evt file."""
        name = self.current_file.name.replace(f"tier_{self.tier}", "tier_raw")
        return self._raw_root() / self.datatype / self.period / self.run / name

    def _open(self, path):
        """A kept-open, read-only handle for ``path``.

        ``lh5.read`` opens and closes the file on every call, and the
        all-waveforms view makes one call per channel -- ~100 opens per
        render, each a metadata round-trip on a network filesystem.

        Per session and never shared: HDF5 handles are not fork-safe, and a
        handle is only ever used from its own session's callbacks, which
        :class:`leds.app.EventDisplay` serialises under one lock even when
        Panel runs them on a thread pool (see "Threads" in the README).
        """
        path = str(path)
        handle = self._open_files.pop(path, None)  # re-inserted below (LRU order)
        if handle is None:
            handle = h5py.File(path, "r", locking=False)
        self._open_files[path] = handle
        while len(self._open_files) > MAX_OPEN_FILES:
            self._open_files.pop(next(iter(self._open_files))).close()
        return handle

    def read_waveform(self, rawid, hit_idx, kind="waveform_windowed"):
        """Return ``(times_ns, values)`` for one detector's raw waveform."""
        wf = lh5.read(
            f"ch{rawid:07d}/raw/{kind}",
            self._open(self.raw_file()),
            start_row=hit_idx,
            n_rows=1,
        )
        values = wf.values.nda[0]
        t0 = float(wf.t0.nda[0])
        dt = float(wf.dt.nda[0])
        times = t0 + dt * np.arange(len(values))
        return times, values
