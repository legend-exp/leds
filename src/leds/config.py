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


def list_cycles(base_path: str | os.PathLike | None = None) -> list[str]:
    """Subdirectories of ``base_path`` that are production cycles.

    A production cycle is a directory containing a ``dataflow-config.yaml``.
    """
    root = resolve_base_path(base_path)
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / CONFIG_FILENAME).is_file()
    )


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
