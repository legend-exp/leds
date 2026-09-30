from __future__ import annotations

from types import SimpleNamespace

from leds.event_viewer import _valid_files


def fake_db(path, entries=None):
    """A TextDB stand-in: only ``__path__`` and its ``validity.yaml`` matter."""
    path.mkdir(parents=True, exist_ok=True)
    if entries is not None:
        lines = []
        for valid_from, mode, files in entries:
            lines += [f"- valid_from: {valid_from}", f"  mode: {mode}", "  apply:"]
            lines += [f"  - {f}" for f in files]
        (path / "validity.yaml").write_text("\n".join(lines) + "\n")
    return SimpleNamespace(__path__=str(path))


def test_one_key_per_validity_entry(tmp_path):
    db = fake_db(
        tmp_path / "channelmaps",
        [
            ("20250101T000000Z", "reset", ["a.yaml"]),
            ("20250201T000000Z", "append", ["b.yaml"]),
        ],
    )

    jan = _valid_files(db, "20250105T120000Z")
    assert jan == _valid_files(db, "20250130T000000Z")  # same entry, same key
    assert jan == ("a.yaml",)
    assert _valid_files(db, "20250202T000000Z") == ("a.yaml", "b.yaml")


def test_no_key_without_a_single_validity_file(tmp_path):
    assert _valid_files(fake_db(tmp_path / "none"), "20250105T120000Z") is None

    both = fake_db(tmp_path / "both", [("20250101T000000Z", "reset", ["a.yaml"])])
    (tmp_path / "both" / "validity.jsonl").write_text("")
    assert _valid_files(both, "20250105T120000Z") is None


def test_no_key_for_a_broken_catalog(tmp_path):
    db = fake_db(tmp_path / "broken")
    (tmp_path / "broken" / "validity.yaml").write_text("- valid_from: [oops\n")
    assert _valid_files(db, "20250105T120000Z") is None
