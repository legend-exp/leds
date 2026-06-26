from __future__ import annotations

import os
from pathlib import Path

from dbetto import AttrsDict, Props

#: Name of the dataflow config file expected at the root of a production cycle.
CONFIG_FILENAME = "dataflow-config.yaml"

#: Environment variable used to point the app at a production cycle when no
#: explicit ``base_path`` is given. Lets the hosted (Docker/spin) instance and a
#: local user select their data without code changes.
ENV_BASE_PATH = "LEDS_BASE_PATH"


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


def discover_cycles(
    base_path: str | os.PathLike | list | None = None,
) -> dict[str, Path]:
    """Map ``label -> cycle directory`` across one or more base paths.

    Each base path is scanned for sub-directories holding a
    ``dataflow-config.yaml``; a base path that is itself a cycle (has the config
    at its root) is included directly. Labels are the cycle directory names,
    qualified with the parent directory name only when two cycles would
    otherwise collide.
    """
    cycles: dict[str, Path] = {}
    for root in resolve_base_paths(base_path):
        found = sorted(
            p for p in root.iterdir() if p.is_dir() and (p / CONFIG_FILENAME).is_file()
        )
        if not found and (root / CONFIG_FILENAME).is_file():
            found = [root]  # the base path is itself a single cycle
        for cdir in found:
            label = cdir.name
            if label in cycles and cycles[label] != cdir:
                label = f"{cdir.parent.name}/{cdir.name}"
            cycles[label] = cdir
    return cycles


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
