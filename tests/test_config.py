from __future__ import annotations

from leds.config import CONFIG_FILENAME, cycle_groups, discover_cycles


def make_cycle(path):
    path.mkdir(parents=True)
    (path / CONFIG_FILENAME).write_text("paths: {}\n")
    return path


def test_cycles_ordered_ref_tmp_auto_newest_first(tmp_path):
    prod = tmp_path / "prod-blind"
    for rel in (
        "ref/v2.0.0",
        "ref/v2.10.0",
        "ref/v2.9.1",
        "tmp/v2.1.0dev1",
        "tmp/v2.1.0",
        "auto/latest",
    ):
        make_cycle(prod / rel)
    # listed in another order than the dropdown's, to prove it is re-sorted
    roots = [prod / "auto", prod / "tmp", prod / "ref"]

    cycles = discover_cycles(roots)

    assert list(cycles) == [
        "ref/v2.10.0",
        "ref/v2.9.1",
        "ref/v2.0.0",
        "tmp/v2.1.0dev1",
        "tmp/v2.1.0",
        "auto/latest",
    ]
    assert cycles["tmp/v2.1.0"] == prod / "tmp" / "v2.1.0"
    assert cycle_groups(cycles) == {
        "ref": {
            "v2.10.0": "ref/v2.10.0",
            "v2.9.1": "ref/v2.9.1",
            "v2.0.0": "ref/v2.0.0",
        },
        "tmp": {"v2.1.0dev1": "tmp/v2.1.0dev1", "v2.1.0": "tmp/v2.1.0"},
        "auto": {"latest": "auto/latest"},
    }


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
    assert cycle_groups(cycles) == {
        "ref": {"v1": "ref/v1"},
        "other": {"mock_prod": "mock_prod"},
    }


def test_plain_cycles_keep_a_plain_dropdown(tmp_path):
    single = make_cycle(tmp_path / "mock_prod")
    assert discover_cycles(single) == {"mock_prod": single}
    assert cycle_groups(discover_cycles(single)) is None
