from __future__ import annotations

import os
import re
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


def _natural_key(name: str) -> list:
    # "v2.10.0" after "v2.9.0": digit runs compare as numbers (re.split with a
    # group alternates str/int from a str, so positions always compare alike)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


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
    two would collide. Ordered by kind (``CYCLE_KINDS``, then unclassified),
    newest first within each, so the first is the newest ref cycle.
    """
    cycles: dict[str, Path] = {}
    for root in resolve_base_paths(base_path):
        found = sorted(
            p for p in root.iterdir() if p.is_dir() and (p / CONFIG_FILENAME).is_file()
        )
        if not found and (root / CONFIG_FILENAME).is_file():
            found = [root]  # the base path is itself a single cycle
        for cdir in found:
            kind = cycle_kind(cdir)
            label = f"{kind}/{cdir.name}" if kind else cdir.name
            if label in cycles and cycles[label] != cdir:
                label = f"{cdir.parent.name}/{cdir.name}"
                if kind:
                    label = f"{cdir.parent.parent.name}/{label}"
            cycles[label] = cdir

    def kind_rank(item):
        kind = cycle_kind(item[1])
        return CYCLE_KINDS.index(kind) if kind else len(CYCLE_KINDS)

    items = sorted(
        cycles.items(), key=lambda kv: _natural_key(kv[1].name), reverse=True
    )
    items.sort(key=kind_rank)  # stable: stays newest first within a kind
    return dict(items)


def cycle_groups(cycles: dict[str, Path]) -> dict[str, dict[str, str]] | None:
    """Dropdown sections ``{kind: {shown name: label}}`` for ``cycles``.

    In the order of ``cycles`` (see :func:`discover_cycles`), with the
    unclassified ones under "other". ``None`` when no cycle has a kind, so a
    plain local cycle keeps a plain dropdown.
    """
    groups: dict[str, dict[str, str]] = {}
    for label, path in cycles.items():
        kind = cycle_kind(path)
        shown = (
            label.split("/", 1)[1] if kind and label.startswith(f"{kind}/") else label
        )
        groups.setdefault(kind or "other", {})[shown] = label
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
