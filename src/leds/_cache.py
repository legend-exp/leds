"""Process-wide caches for read-only data shared by every session.

The dashboard builds one :class:`~leds.event_viewer.EventViewer` (and its
metadata, channelmaps, directory scans and per-run reductions) *per browser
session*. Almost all of that is a pure function of what is on disk, so N users
looking at the same production cycle were paying N times for identical work --
the single largest contributor to session spin-up.

The caches here are keyed only by filesystem-derived strings, never by session
state or by object identity, so an entry built for one session is valid for
every other session pointing at the same cycle. They are bounded (a hosted
container runs for weeks) and, for anything derived from the metadata
checkout, time-limited, so an updated checkout is picked up without a restart.

Deliberately Panel-free: this sits in the data layer, which the view modules
rely on being framework-agnostic. Sizes and lifetimes are tunable by
environment variable -- note every bound is *per worker process*, so the
container's memory ceiling should account for ``--num-procs``.
"""

from __future__ import annotations

import os
import threading
import time

#: Sentinel distinguishing "absent" from a cached ``None``.
_MISS = object()


def _env_int(name, default):
    try:
        return max(0, int(os.environ[name]))
    except (KeyError, ValueError):
        return default


#: Lifetime of anything derived from the metadata checkout (channelmaps,
#: detector statuses, calibration parameters). dbetto's ``TextDB`` never
#: re-reads a file it has already parsed, so without this a worker would serve
#: the metadata as it stood when it started until the container was restarted.
METADATA_TTL = _env_int("LEDS_CACHE_TTL", 3600)

#: Lifetime of directory scans. Short, because new runs land while the server
#: is up and users expect to see them.
SCAN_TTL = _env_int("LEDS_SCAN_TTL", 300)


class SharedLRU:
    """A bounded, optionally time-limited cache shared across sessions.

    ``get(key, factory)`` returns the cached value or builds it. Concurrent
    callers asking for the same missing key build it once and the rest wait,
    which matters when the "build" is a multi-second read of a whole run.
    """

    def __init__(self, max_items, ttl=None, *, name=""):
        self.name = name
        self.max_items = max_items
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self._entries: dict = {}  # key -> (value, stored_at); insertion-ordered
        self._building: dict = {}  # key -> Lock held while a factory runs
        self._lock = threading.Lock()

    def _fresh(self, key):
        """Cached value for ``key``, or ``_MISS``. Caller holds ``self._lock``."""
        entry = self._entries.get(key, _MISS)
        if entry is _MISS:
            return _MISS
        value, stored_at = entry
        if self.ttl is not None and time.monotonic() - stored_at > self.ttl:
            del self._entries[key]
            return _MISS
        # refresh the LRU position
        del self._entries[key]
        self._entries[key] = (value, stored_at)
        return value

    def _store(self, key, value):
        """Caller holds ``self._lock``."""
        self._entries.pop(key, None)
        self._entries[key] = (value, time.monotonic())
        while len(self._entries) > self.max_items:
            self._entries.pop(next(iter(self._entries)))

    def get(self, key, factory):
        with self._lock:
            value = self._fresh(key)
            if value is not _MISS:
                self.hits += 1
                return value
            self.misses += 1
            build_lock = self._building.get(key)
            if build_lock is None:
                build_lock = self._building[key] = threading.Lock()

        with build_lock:
            # another caller may have finished building while we queued
            with self._lock:
                value = self._fresh(key)
                if value is not _MISS:
                    return value
            try:
                value = factory()
            finally:
                with self._lock:
                    self._building.pop(key, None)
            with self._lock:
                self._store(key, value)
            return value

    def peek(self, key, default=None):
        """The cached value for ``key``, or ``default``; never builds one."""
        with self._lock:
            value = self._fresh(key)
        return default if value is _MISS else value

    def clear(self):
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# The shared caches themselves, in one place so the memory budget is reviewable
# ---------------------------------------------------------------------------

#: ``(metadata_path, <files valid at the timestamp>)`` -> channelmap, so every
#: DAQ file of one validity entry shares a build (falls back to the timestamp).
#: The dominant per-session cost: building one parses the whole metadata
#: checkout and shells out to git. Entries are ``AttrsDict`` objects marked
#: read-only all the way down, so a session cannot corrupt another's view.
CHANNELMAPS = SharedLRU(32, ttl=METADATA_TTL, name="channelmaps")

#: ``(validity_path,)`` -> parsed ``Catalog``, to resolve the files valid at a
#: timestamp in microseconds (the cache keys above and below).
CATALOGS = SharedLRU(64, ttl=METADATA_TTL, name="catalogs")

#: ``(metadata_path,)`` -> ``LegendMetadata``. Sharing these also shares their
#: internal ``TextDB`` file store, so runinfo and groupings are parsed once.
METADATA = SharedLRU(8, ttl=METADATA_TTL, name="metadata")

#: ``(status_path, <files valid at start_key>, category)`` -> detector
#: statuses (falls back to the start key). ``TextDB.on`` re-parses
#: ``validity.yaml`` on every call, and the Dataset tab calls it once per run.
STATUSES = SharedLRU(256, ttl=METADATA_TTL, name="statuses")

#: ``(par_root, files)`` -> the calibration part of the ``par_hit``/``par_pht``
#: files ``validity.yaml`` lists for a start key, merged. ~5 MB and ~0.5 s to
#: parse each (the whole file would be ~40 MB and ~10 s). A reprocessing
#: rewrites these files in place under the same name, hence the TTL.
CAL_PARS = SharedLRU(
    _env_int("LEDS_MAX_CACHED_CAL_PARS", 32), ttl=METADATA_TTL, name="cal_pars"
)

#: Directory scans: ``(tier_root,)`` -> run tree, ``(tier_root, period, run)``
#: -> file list. Mutable on disk, hence the short TTL.
RUN_TREES = SharedLRU(16, ttl=SCAN_TTL, name="run_trees")
RUN_FILES = SharedLRU(1024, ttl=SCAN_TTL, name="run_files")

#: ``(file, group)`` -> event count. Written files are immutable, so no TTL.
N_EVENTS = SharedLRU(4096, name="n_events")

#: ``(tier_root, period, run, files)`` -> the run's whole per-hit energy and
#: cut arrays. **The dominant memory line**: hundreds of MB per entry on a real
#: run, so the bound is small and deliberately tunable per deployment. Keyed by
#: the file list, so a run that gains a file is a new entry, not a stale one.
RUN_SPECTRA = SharedLRU(_env_int("LEDS_MAX_CACHED_RUN_SPECTRA", 4), name="run_spectra")

#: ``(tier_root, period, run, files)`` -> binned validation summary of a run,
#: all strings included: ~1.3 MB each. The per-string counts select hits by a
#: rawid -> string map read from the channelmap, so they are metadata-derived
#: and expire with it.
VALIDATION_SUMMARIES = SharedLRU(
    _env_int("LEDS_MAX_CACHED_VALIDATION_RUNS", 64),
    ttl=METADATA_TTL,
    name="validation_summaries",
)

#: Calibration check, per cal run (keyed by the files read): per-detector
#: uncut 0.5 keV spectra (~1.4 MB), and the is_valid_cal section data (~2 MB:
#: a histogram of passing hits plus the few failing hits). 32 runs is about
#: one period. Files are immutable, so no TTL.
CAL_CHECK_SPECTRA = SharedLRU(32, name="cal_check_spectra")
CAL_CHECK_CUTS = SharedLRU(32, name="cal_check_cuts")

#: Every cache above, for the warm-up path and for tests.
ALL = (
    CHANNELMAPS,
    CATALOGS,
    METADATA,
    STATUSES,
    CAL_PARS,
    RUN_TREES,
    RUN_FILES,
    N_EVENTS,
    RUN_SPECTRA,
    VALIDATION_SUMMARIES,
    CAL_CHECK_SPECTRA,
    CAL_CHECK_CUTS,
)


def clear_all():
    """Drop every shared entry (used by tests and on an explicit refresh)."""
    for cache in ALL:
        cache.clear()
