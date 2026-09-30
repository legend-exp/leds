from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
from pathlib import Path

from dbetto import AttrsDict, Props

#: Name of the dataflow config file expected at the root of a production cycle.
CONFIG_FILENAME = "dataflow-config.yaml"

#: Environment variable used to point the app at a production cycle when no
#: explicit ``base_path`` is given. Lets the hosted (Docker/spin) instance and a
#: local user select their data without code changes.
ENV_BASE_PATH = "LEDS_BASE_PATH"

#: Kinds of production cycle, in dropdown order, named by the directory that
#: holds them: ``prod-blind/ref/v2.1.0``, ``prod-blind/tmp/v2.1.0dev1``,
#: ``prod-blind/auto/latest``.
CYCLE_KINDS = ("ref", "tmp", "auto")

#: Event tiers the viewer can open (see ``EventViewer``); a cycle with neither
#: is not listed.
EVENT_TIERS = ("pet", "evt")

# glibc statx(), for birth times: Python has no st_birthtime on Linux
_AT_FDCWD = -100
_AT_SYMLINK_NOFOLLOW = 0x100
_STATX_BTIME = 0x800
_STATX_BTIME_OFFSET = 80  # of stx_btime in struct statx
try:
    _statx = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True).statx
    _statx.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    _statx.restype = ctypes.c_int
except (OSError, AttributeError, TypeError):  # not glibc (e.g. macOS)
    _statx = None


def resolve_base_path(base_path: str | os.PathLike | None = None) -> Path:
    """Resolve the production-cycle base path from an argument or the environment.

    Resolution order: explicit ``base_path`` argument, then ``$LEDS_BASE_PATH``.
    """
    if base_path is None:
        base_path = os.environ.get(ENV_BASE_PATH)
    if not base_path:
        msg = (
            "no production cycle given: pass base_path or set "
            f"${ENV_BASE_PATH} to the directory containing {CONFIG_FILENAME}"
        )
        raise ValueError(msg)

    path = Path(base_path).expanduser()
    if not path.is_dir():
        msg = f"base path is not a directory: {path}"
        raise FileNotFoundError(msg)
    return path


def resolve_base_paths(
    base_path: str | os.PathLike | list | None = None,
) -> list[Path]:
    """Resolve one or more production-cycle base paths.

    Accepts a single path, an iterable of paths, or (when ``None``) the
    ``$LEDS_BASE_PATH`` environment variable. A string -- including the env var
    -- may list several directories separated by ``os.pathsep`` (``:`` on Unix).
    Duplicates are removed, order preserved; each must be an existing directory.
    """
    if isinstance(base_path, (str, os.PathLike)):
        items = str(base_path).split(os.pathsep)
    elif base_path:  # non-empty iterable of paths
        items = list(base_path)
    else:  # None or empty -> environment
        items = os.environ.get(ENV_BASE_PATH, "").split(os.pathsep)

    items = [s for s in (str(i).strip() for i in items) if s]
    if not items:
        msg = (
            "no production cycle given: pass base_path(s) or set "
            f"${ENV_BASE_PATH} to one or more directories (separated by "
            f"{os.pathsep!r}) containing {CONFIG_FILENAME}"
        )
        raise ValueError(msg)

    paths: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        if not path.is_dir():
            msg = f"base path is not a directory: {path}"
            raise FileNotFoundError(msg)
        if path not in paths:
            paths.append(path)
    return paths


def list_cycles(base_path: str | os.PathLike | None = None) -> list[str]:
    """Subdirectories of ``base_path`` that are production cycles.

    A production cycle is a directory containing a ``dataflow-config.yaml``.
    """
    root = resolve_base_path(base_path)
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / CONFIG_FILENAME).is_file()
    )


def cycle_kind(path: str | os.PathLike) -> str | None:
    """The cycle's kind (one of ``CYCLE_KINDS``) from its parent directory."""
    kind = Path(path).parent.name
    return kind if kind in CYCLE_KINDS else None


def _statx_birthtime(path: Path) -> float | None:
    """``path``'s own birth time from ``statx``, ``None`` if not reported."""
    if _statx is None:
        return None
    buf = ctypes.create_string_buffer(256)
    flags = _AT_SYMLINK_NOFOLLOW
    if _statx(_AT_FDCWD, os.fsencode(path), flags, _STATX_BTIME, buf) != 0:
        return None
    (mask,) = struct.unpack_from("I", buf, 0)
    if not mask & _STATX_BTIME:
        return None
    sec, nsec = struct.unpack_from("qI", buf, _STATX_BTIME_OFFSET)
    return sec + nsec / 1e9


def _created(path: str | os.PathLike) -> float:
    """When ``path`` itself was created (for a symlink: the link, not its target).

    The birth time where the filesystem reports it. Otherwise a symlink's own
    mtime (links are never modified), or the cycle config's mtime (a cycle
    directory's own mtime changes with its contents).
    """
    path = Path(path)
    st = os.lstat(path)
    born = getattr(st, "st_birthtime", None) or _statx_birthtime(path)
    if born:
        return born
    if path.is_symlink():
        return st.st_mtime
    return (path / CONFIG_FILENAME).stat().st_mtime


def has_event_tier(cycle_dir: str | os.PathLike, datatype: str = "phy") -> bool:
    """Whether the cycle has an event tier (``EVENT_TIERS``) the viewer can open."""
    try:
        paths = load_paths(cycle_dir)
        return any(
            f"tier_{t}" in paths and (Path(paths[f"tier_{t}"]) / datatype).is_dir()
            for t in EVENT_TIERS
        )
    except Exception:  # unreadable or half-written config
        return False


def discover_cycles(
    base_path: str | os.PathLike | list | None = None,
) -> dict[str, Path]:
    """Map ``label -> cycle directory`` across one or more base paths.

    Each base path is scanned for sub-directories holding a
    ``dataflow-config.yaml``; a base path that is itself a cycle (has the config
    at its root) is included directly.

    A cycle inside a ``ref``/``tmp``/``auto`` directory is labelled
    ``<kind>/<name>`` (the same version can exist as both ref and tmp); any
    other by its directory name, qualified with the parent's name only when
    two would collide. Cycles without an event tier (see :func:`has_event_tier`)
    are left out.

    Ordered by kind (``CYCLE_KINDS``, then unclassified); within a kind,
    symlinks (``auto/latest``, ``tmp/p19+``) come first, then the cycles
    themselves, each newest first by creation time (:func:`_created`).
    """
    cycles: dict[str, Path] = {}
    for root in resolve_base_paths(base_path):
        found = sorted(
            p for p in root.iterdir() if p.is_dir() and (p / CONFIG_FILENAME).is_file()
        )
        if not found and (root / CONFIG_FILENAME).is_file():
            found = [root]  # the base path is itself a single cycle
        for cdir in filter(has_event_tier, found):
            kind = cycle_kind(cdir)
            label = f"{kind}/{cdir.name}" if kind else cdir.name
            if label in cycles and cycles[label] != cdir:
                label = f"{cdir.parent.name}/{cdir.name}"
                if kind:
                    label = f"{cdir.parent.parent.name}/{label}"
            cycles[label] = cdir

    def order(item):
        label, path = item
        kind = cycle_kind(path)
        try:
            created = _created(path)
        except OSError:  # vanished while scanning
            created = 0.0
        rank = CYCLE_KINDS.index(kind) if kind else len(CYCLE_KINDS)
        return rank, not path.is_symlink(), -created, label

    return dict(sorted(cycles.items(), key=order))


def cycle_groups(cycles: dict[str, Path]) -> dict[str, list[str]] | None:
    """Dropdown sections ``{kind: [label, ...]}`` for ``cycles``.

    In the order of ``cycles`` (see :func:`discover_cycles`), with the
    unclassified ones under "other". Options keep their full ``ref/v2.1.0``
    label so the closed dropdown still says which kind is selected. ``None``
    when no cycle has a kind, so a plain local cycle keeps a plain dropdown.
    """
    groups: dict[str, list[str]] = {}
    for label, path in cycles.items():
        groups.setdefault(cycle_kind(path) or "other", []).append(label)
    return None if set(groups) <= {"other"} else groups


def load_paths(base_path: str | os.PathLike | None = None) -> AttrsDict:
    """Load the ``paths`` table from a production cycle's ``dataflow-config.yaml``.

    Path variables (``$_``) are substituted relative to the config file, matching
    the legend-dataflow / monitor-dashboard convention.
    """
    base = resolve_base_path(base_path)
    cfg_file = base / CONFIG_FILENAME
    if not cfg_file.is_file():
        msg = f"no {CONFIG_FILENAME} found in {base}"
        raise FileNotFoundError(msg)

    prod_config = AttrsDict(Props.read_from(str(cfg_file), subst_pathvar=True))
    return prod_config.paths
