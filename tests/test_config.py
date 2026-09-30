from __future__ import annotations

import os
import time

import pytest

from leds import config
from leds.config import CONFIG_FILENAME, cycle_groups, discover_cycles


def make_cycle(path, *, tier="evt"):
    """A cycle directory; ``tier=None`` gives one without an event tier."""
    path.mkdir(parents=True)
    paths = f"{{tier_{tier}: $_/generated/tier/{tier}}}" if tier else "{}"
    (path / CONFIG_FILENAME).write_text(f"paths: {paths}\n")
    if tier:
        (path / "generated" / "tier" / tier / "phy").mkdir(parents=True)
    return path


def fake_created(monkeypatch, root, times):
    """Creation times by path relative to ``root``: later in ``times`` = newer."""
    rank = {rel: i for i, rel in enumerate(times)}
    monkeypatch.setattr(config, "_created", lambda p: rank[os.path.relpath(p, root)])


def test_symlinks_first_then_cycles_each_newest_first(tmp_path, monkeypatch):
    prod = tmp_path / "prod-blind"
    for rel in ("ref/v2.0.0", "ref/v2.10.0", "tmp/v2.1.0dev1", "auto/v1.0.0"):
        make_cycle(prod / rel)
    (prod / "ref" / "napoli26").symlink_to("v2.0.0")
    (prod / "ref" / "latest").symlink_to("v2.10.0")
    (prod / "auto" / "latest").symlink_to("v1.0.0")
    fake_created(
        monkeypatch,
        prod,
        # oldest first; note v2.0.0 was created after v2.10.0
        [
            "ref/v2.10.0",
            "auto/v1.0.0",
            "ref/v2.0.0",
            "ref/latest",
            "tmp/v2.1.0dev1",
            "ref/napoli26",
            "auto/latest",
        ],
    )
    # listed in another order than the dropdown's, to prove it is re-sorted
    roots = [prod / "auto", prod / "tmp", prod / "ref"]

    cycles = discover_cycles(roots)

    assert list(cycles) == [
        "ref/napoli26",
        "ref/latest",
        "ref/v2.0.0",
        "ref/v2.10.0",
        "tmp/v2.1.0dev1",
        "auto/latest",
        "auto/v1.0.0",
    ]
    assert cycles["ref/napoli26"] == prod / "ref" / "napoli26"
    assert cycle_groups(cycles) == {
        "ref": ["ref/napoli26", "ref/latest", "ref/v2.0.0", "ref/v2.10.0"],
        "tmp": ["tmp/v2.1.0dev1"],
        "auto": ["auto/latest", "auto/v1.0.0"],
    }


def test_cycles_without_an_event_tier_are_hidden(tmp_path):
    make_cycle(tmp_path / "ref" / "v3.0.0", tier=None)  # raw only
    pet = make_cycle(tmp_path / "ref" / "v3.3.0", tier="pet")
    (tmp_path / "ref" / "raw").symlink_to("v3.0.0")
    broken = tmp_path / "ref" / "half-written"
    broken.mkdir()
    (broken / CONFIG_FILENAME).write_text("paths: [unclosed\n")

    assert discover_cycles(tmp_path / "ref") == {"ref/v3.3.0": pet}


@pytest.mark.parametrize("btime", [True, False])
def test_created_is_the_symlinks_own_time(tmp_path, monkeypatch, btime):
    if not btime:  # a filesystem without birth times: the mtime fallbacks
        monkeypatch.setattr(config, "_statx_birthtime", lambda _p: None)
    target = make_cycle(tmp_path / "ref" / "v1")
    old = time.time() - 10 * 86400
    os.utime(target / CONFIG_FILENAME, (old, old))
    link = tmp_path / "ref" / "latest"
    link.symlink_to("v1")

    # a link made now is newer than its target, whatever the target's times
    assert config._created(link) >= config._created(target)
    assert config._created(link) > old


def test_same_version_as_ref_and_tmp_does_not_collide(tmp_path):
    ref = make_cycle(tmp_path / "ref" / "v2.1.0")
    tmp = make_cycle(tmp_path / "tmp" / "v2.1.0")

    cycles = discover_cycles([tmp_path / "ref", tmp_path / "tmp"])

    assert cycles == {"ref/v2.1.0": ref, "tmp/v2.1.0": tmp}


def test_unclassified_cycles_follow_the_kinds_under_other(tmp_path):
    make_cycle(tmp_path / "prod" / "ref" / "v1")
    bare = make_cycle(tmp_path / "mock_prod")

    cycles = discover_cycles([tmp_path / "prod" / "ref", bare])

    assert list(cycles) == ["ref/v1", "mock_prod"]
    assert cycle_groups(cycles) == {"ref": ["ref/v1"], "other": ["mock_prod"]}


def test_plain_cycles_keep_a_plain_dropdown(tmp_path):
    single = make_cycle(tmp_path / "mock_prod")
    assert discover_cycles(single) == {"mock_prod": single}
    assert cycle_groups(discover_cycles(single)) is None
