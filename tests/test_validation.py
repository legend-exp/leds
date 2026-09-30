from __future__ import annotations

from types import SimpleNamespace

import awkward as ak
import boost_histogram as bh
import numpy as np
import pytest
import yaml

from leds import _cache, validation_view
from leds import validation as validation_mod
from leds.validation import (
    BIN_WIDTHS,
    K_LINES,
    ValidationData,
    _read_cal_yaml,
    cal_curve,
    cal_residuals,
    qc_flag_table,
    survival_fraction,
)

DAY = 86400
T0 = 200 * DAY  # an arbitrary absolute day boundary


class FakeViewer:
    group = "evt"

    def __init__(self, runs=None, paths=None, runinfo=None):
        self._runs = runs or {}
        self.paths = paths or {}
        self.status_db = SimpleNamespace(runinfo=runinfo or {})
        # distinct per instance, so two fakes never share a shared-cache entry
        self.cycle_key = f"fake-viewer-{id(self)}"

    def available_runs(self):
        return self._runs

    def _run_files(self, _period, _run):
        return []

    def _run_tstamps(self, period, run):
        return list(self._runs.get(period, {}).get(run, []))


def columns(t, **overrides):
    """Synthetic evt columns: all-physics events unless overridden."""
    n = len(t)
    cols = {
        "timestamp": np.asarray(t, dtype=float),
        "forced": np.zeros(n, dtype=bool),
        "puls": np.zeros(n, dtype=bool),
        "muon": np.zeros(n, dtype=bool),
        "muon_offline": np.zeros(n, dtype=bool),
        "spms": np.zeros(n, dtype=bool),
        "mult": np.ones(n, dtype=np.uint16),
        "qc": np.ones(n, dtype=bool),
        "energy": ak.Array([[] for _ in range(n)]),
        "psd_bb": ak.Array([[] for _ in range(n)]),
    }
    cols.update(overrides)
    return cols


def make_data(columns_by_run, runs=None):
    """A ValidationData whose _columns serves the given synthetic columns."""
    runs = runs or {"p01": {run: ["ts"] for run in sorted(columns_by_run)}}
    data = ValidationData(FakeViewer(runs=runs))
    data._columns = lambda _period, run: columns_by_run[run]  # type: ignore[method-assign]
    # unit mass so rate expectations stay in plain Hz / counts-per-hour
    data._mass_kg = lambda _period, _run, _string=None: 1.0  # type: ignore[method-assign]
    return data


def counts_of(data, period, run, key):
    return data._summary(period, run)["series"][key].view().sum()


# ---------------------------------------------------------------- binning


@pytest.mark.parametrize("label", list(BIN_WIDTHS))
def test_constant_rate_flat_at_every_width(label):
    # one event every 10 s for 6 h -> 0.1 Hz at any bin width, incl. edge bins
    t = np.arange(T0 + 7200, T0 + 7200 + 6 * 3600, 10.0)
    data = make_data({"r001": columns(t)})
    times, rates = data.period_series("p01", BIN_WIDTHS[label])

    assert times.size > 0
    rate = rates[("trigger", "all triggers")]
    np.testing.assert_allclose(rate, 0.1, rtol=0.02)
    # total counts are preserved by the exposure-weighted rebinning
    summary = data._summary("p01", "r001")
    assert counts_of(data, "p01", "r001", ("trigger", "all triggers")) == t.size
    # base axis is day-aligned so every configured width divides it exactly
    edges = summary["exposure"].axes[0].edges
    assert edges[0] % DAY == 0
    assert edges[-1] % DAY == 0
    assert (edges[-1] - edges[0]) % BIN_WIDTHS[label] == 0


def test_rebin_needs_no_reread():
    t = np.arange(T0, T0 + 3600, 5.0)
    data = make_data({"r001": columns(t)})
    data.period_series("p01", 900)
    data._columns = None  # any further read attempt would raise
    _times, rates = data.period_series("p01", 3600)  # served from the cache
    assert np.isfinite(rates[("trigger", "all triggers")]).any()


# ---------------------------------------------------------------- baselines


def test_forced_and_pulser_excluded_from_physics_series():
    n = 100
    t = T0 + np.arange(n, dtype=float)
    forced = np.zeros(n, dtype=bool)
    puls = np.zeros(n, dtype=bool)
    forced[:20] = True
    puls[20:30] = True
    data = make_data({"r001": columns(t, forced=forced, puls=puls)})

    assert counts_of(data, "p01", "r001", ("trigger", "all triggers")) == 100
    assert counts_of(data, "p01", "r001", ("trigger", "forced")) == 20
    assert counts_of(data, "p01", "r001", ("trigger", "pulser")) == 10
    # multiplicity and qc series count only the 70 physics events
    assert counts_of(data, "p01", "r001", ("multiplicity", "m = 1")) == 70
    assert counts_of(data, "p01", "r001", ("qc", "pass")) == 70
    assert counts_of(data, "p01", "r001", ("qc", "fail")) == 0


def test_kline_cut_configs():
    k40 = K_LINES["K40"]
    k42 = K_LINES["K42"]
    t = T0 + np.arange(5, dtype=float)
    cols = columns(
        t,
        energy=ak.Array([[k40], [k42], [k40], [800.0], [k40, 700.0]]),
        psd_bb=ak.Array([[True], [False], [True], [True], [False, True]]),
        qc=np.array([True, False, True, True, True]),
        mult=np.array([1, 2, 1, 1, 2], dtype=np.uint16),
        spms=np.array([False, True, False, False, False]),
        forced=np.array([False, False, True, False, False]),
    )
    data = make_data({"r001": cols})

    def k(line, config):
        return counts_of(data, "p01", "r001", (line, config))

    # event 2 is forced -> excluded everywhere; event 4's only K40 hit fails psd
    assert k("K40", "before cuts") == 2
    assert k("K40", "after QC") == 2
    assert k("K40", "after mult = 1") == 1
    assert k("K40", "after LAr") == 2
    assert k("K40", "after PSD") == 1
    # event 1 fails qc, mult and LAr, and its hit fails psd
    assert k("K42", "before cuts") == 1
    assert k("K42", "after QC") == 0
    assert k("K42", "after mult = 1") == 0
    assert k("K42", "after LAr") == 0
    assert k("K42", "after PSD") == 0


def test_missing_fields_disable_series_only():
    t = T0 + np.arange(10, dtype=float)
    cols = columns(t, mult=None, muon_offline=None, energy=None, psd_bb=None)
    data = make_data({"r001": cols})
    times, rates = data.period_series("p01", 3600)

    assert times.size > 0
    assert rates[("trigger", "muon offline")] is None
    assert all(rates[("multiplicity", m)] is None for m in ("m = 0", "m = 1"))
    assert rates[("K40", "before cuts")] is None
    assert np.isfinite(rates[("trigger", "all triggers")]).any()
    assert np.isfinite(rates[("qc", "pass")]).any()


def test_string_scoping_and_mass_normalisation():
    k40 = K_LINES["K40"]
    t = T0 + np.array([0.0, 600.0, 1200.0, 1800.0])
    cols = columns(
        t,
        rawid=ak.Array([[101], [201], [101, 201], []]),
        energy=ak.Array([[k40], [k40], [800.0, k40], []]),
        psd_bb=ak.Array([[True], [True], [True, True], []]),
        mult=np.array([1, 1, 2, 0], dtype=np.uint16),
    )
    data = ValidationData(FakeViewer(runs={"p01": {"r001": ["ts"]}}))
    data._columns = lambda _period, _run: cols  # type: ignore[method-assign]
    data._string_map = lambda _period, _run: (  # type: ignore[method-assign]
        {1: frozenset({101}), 2: frozenset({201})},
        {1: 2.0, 2: 4.0},
    )

    def n(string, key):
        return data._summary("p01", "r001")["strings"][key][:, bh.loc(string)].sum()

    # events count toward a string only when they have a hit in it
    assert n(1, ("trigger", "all triggers")) == 2  # e0, e2
    assert n(2, ("trigger", "all triggers")) == 2  # e1, e2
    assert n(1, ("multiplicity", "m = 1")) == 1
    assert n(1, ("multiplicity", "m = 2")) == 1
    assert n(1, ("multiplicity", "m = 0")) == 0  # m=0 events have no hits
    # the in-window hit must be in the string: e2's K40 hit is in string 2
    assert n(1, ("K40", "before cuts")) == 1
    assert n(2, ("K40", "before cuts")) == 2

    assert data._mass_kg("p01", "r001", 1) == 2.0
    assert data._mass_kg("p01", "r001") == 6.0
    assert data._mass_kg("p01", "r001", 9) == 0.0
    assert data.available_strings("p01") == [1, 2]

    # rates are divided by the scope's mass: (2 evts / 2 kg) / (4 evts / 6 kg)
    _, r1 = data.period_series("p01", 3600, string=1)
    _, rall = data.period_series("p01", 3600)
    ratio = r1[("trigger", "all triggers")] / rall[("trigger", "all triggers")]
    np.testing.assert_allclose(ratio[~np.isnan(ratio)], 1.5)


def test_qc_survival_by_string():
    t = T0 + np.array([0.0, 600.0, 1200.0, 1800.0])
    cols = columns(
        t,
        rawid=ak.Array([[101], [201], [101, 201], [201]]),
        qc=np.array([True, False, True, True]),
    )
    data = ValidationData(FakeViewer(runs={"p01": {"r001": ["ts"]}}))
    data._columns = lambda _period, _run: cols  # type: ignore[method-assign]
    data._string_map = lambda _period, _run: (  # type: ignore[method-assign]
        {1: frozenset({101}), 2: frozenset({201})},
        {1: 2.0, 2: 4.0},
    )

    times, fracs, _ = data.qc_survival_by_string("p01", 3600)

    assert set(fracs) == {1, 2}
    np.testing.assert_allclose(fracs[1][~np.isnan(fracs[1])], 1.0)  # e0, e2 pass
    np.testing.assert_allclose(fracs[2][~np.isnan(fracs[2])], 2 / 3)  # e1 fails
    assert times.size == fracs[1].size


class FakeLGDO:
    """Just enough of an lgdo for the QC flag reduction."""

    def __init__(self, value):
        self.value = value
        self.nda = None if isinstance(value, ak.Array) else np.asarray(value)

    def view_as(self, _kind):
        return self.value


def test_qc_flag_counts_and_names(monkeypatch):
    nbb = "geds/quality/is_not_bb_like"
    raw = {
        "trigger/is_forced": FakeLGDO([False, False, True]),
        "coincident/puls": FakeLGDO([False, False, False]),
        f"{nbb}/rawid": FakeLGDO(ak.Array([[101, 201], [101], [101]])),
        # bit 0 = flag_a, bit 1 = flag_b; set = passed
        f"{nbb}/is_empty_bits": FakeLGDO(ak.Array([[0b10, 0b00], [0b01], [0b00]])),
        f"{nbb}/is_delayed_discharge": FakeLGDO([True, False, True]),
        # stored as all-NaN floats in some cycles: carries no information
        f"{nbb}/is_pos_polarity_bits": FakeLGDO(
            ak.Array([[np.nan, np.nan], [np.nan], [np.nan]])
        ),
    }
    monkeypatch.setattr(validation_mod, "_read_groups", lambda *_a, **_k: raw)
    data = ValidationData(FakeViewer(runs={"p01": {"r001": ["ts"]}}))

    counts = data.period_qc_flags("p01")

    assert counts["events"] == 2  # the forced event is left out
    assert counts["failing"] == {101: 2, 201: 1}
    assert counts["discharge"] == 1
    assert set(counts["bits"]) == {"is_empty_bits"}  # the NaN class is left out
    flags, table = qc_flag_table(counts, {"is_empty_bits": ["flag_a", "flag_b"]})
    assert flags == ["flag_a", "flag_b"]
    assert table[101] == {"flag_a": 1, "flag_b": 1}
    assert table[201] == {"flag_a": 1, "flag_b": 1}
    # a table narrower than the stored bits gets numbered labels instead
    flags, _ = qc_flag_table(counts, {"is_empty_bits": ["only_one"]})
    assert flags == ["is_empty bit 0", "is_empty bit 1"]


def test_light_summaries_match_the_full_ones():
    t = T0 + np.arange(0, 3 * 3600, 7.0)
    n = t.size
    rng = np.random.default_rng(3)
    cols = columns(
        t,
        forced=rng.random(n) < 0.1,
        puls=rng.random(n) < 0.05,
        muon=rng.random(n) < 0.02,
        mult=rng.integers(0, 4, n).astype(np.uint16),
        qc=rng.random(n) < 0.9,
    )
    heavy = ("rawid", "energy", "psd_bb", "spms")
    data = make_data({"r001": cols})
    data._light_columns = lambda _period, _run: cols | dict.fromkeys(heavy)  # type: ignore[method-assign]
    keys = [
        k
        for k in validation_mod.RATE_GROUPS.items()
        if k[0] in validation_mod.LIGHT_GROUPS
    ]
    keys = [(g, label) for g, labels in keys for label in labels]

    light = data._summary("p01", "r001", light=True)
    validation_mod.VALIDATION_SUMMARIES.clear()  # not served from the full one
    full = data._summary("p01", "r001")

    for k in keys:
        np.testing.assert_array_equal(
            light["series"][k].view(), full["series"][k].view()
        )
    assert light["strings"] is None


def test_period_series_so_far_fills_in_without_waiting():
    runs = {
        f"r00{i}": columns(T0 + i * DAY + np.arange(0, 3600, 10.0)) for i in range(3)
    }
    data = make_data(runs)
    data._light_columns = lambda _period, run: runs[run]  # type: ignore[method-assign]
    keys = [("trigger", "all triggers")]

    data.period_series_so_far("p01", 3600, keys=keys)  # queues all three runs
    _cache._LATER.submit(lambda: None).result(timeout=10)  # let them build
    times, rates, (built, total, errors) = data.period_series_so_far(
        "p01", 3600, keys=keys
    )

    assert (built, total, errors) == (3, 3, [])
    ref_times, ref_rates = data.period_series("p01", 3600, keys=keys)
    np.testing.assert_array_equal(times, ref_times)
    np.testing.assert_array_equal(rates[keys[0]], ref_rates[keys[0]])


def test_period_series_rejects_bad_bin_seconds():
    data = make_data({"r001": columns(T0 + np.arange(0.0, 60.0, 10.0))})
    for bad in (0, 450, 900 + 1, -3600):
        with pytest.raises(ValueError, match="multiple"):
            data.period_series("p01", bad)


def test_survival_fraction():
    p = np.array([9.0, 0.0, 0.0, np.nan])
    f = np.array([1.0, 2.0, 0.0, np.nan])
    out = survival_fraction(p, f)
    np.testing.assert_allclose(out[:2], [0.9, 0.0])
    assert np.isnan(out[2])  # empty bin
    assert np.isnan(out[3])  # run-gap separator
    assert survival_fraction(None, f) is None
    assert survival_fraction(p, None) is None


# ---------------------------------------------------------------- period assembly


def test_period_concatenation_with_gap():
    t1 = T0 + np.arange(0, 3600, 10.0)
    t2 = T0 + 3 * DAY + np.arange(0, 3600, 10.0)
    data = make_data({"r001": columns(t1), "r002": columns(t2)})
    times, rates = data.period_series("p01", 3600)

    assert np.all(np.diff(times) > 0)
    rate = rates[("trigger", "all triggers")]
    assert rate.size == times.size
    assert np.isnan(rate).sum() == 1  # exactly the one separator row
    np.testing.assert_allclose(rate[~np.isnan(rate)], 0.1, rtol=0.02)


def test_empty_period():
    data = make_data({}, runs={"p01": {}})
    times, rates = data.period_series("p01", 3600)
    assert times.size == 0
    assert all(v is None for v in rates.values())


# ---------------------------------------------------------------- calibration


def synthetic_pars(b=0.5):
    pk = {
        2614.511: {
            "validity": True,
            "parameters": {"mu": 2614.511 / b},
            "uncertainties": {"mu": 2.0},
        },
        583.191: {
            "validity": False,  # invalid -> skipped
            "parameters": {"mu": 583.191 / b},
            "uncertainties": {"mu": 1.0},
        },
        "1000.0": {  # string peak keys must work too
            "validity": True,
            "parameters": {"mu": 1000.0 / b},
            "uncertainties": {"mu": 1.0},
        },
    }
    det = {
        "pars": {
            "operations": {
                "cuspEmax_ctc_cal": {
                    "expression": "a + b * cuspEmax_ctc",
                    "parameters": {"a": 0.0, "b": b},
                }
            }
        },
        "results": {"ecal": {"cuspEmax_ctc_cal": {"pk_fits": pk}}},
    }
    return {"V01": det}


def test_cal_curve_linear():
    curve = cal_curve(synthetic_pars(), "V01")
    np.testing.assert_allclose(curve["peaks"], [1000.0, 2614.511])
    np.testing.assert_allclose(curve["cal_mu"], curve["peaks"])  # perfect cal
    np.testing.assert_allclose(curve["residual"], 0.0, atol=1e-9)
    np.testing.assert_allclose(curve["cal_err"], 0.5 * curve["mu_err"])
    np.testing.assert_allclose(curve["line_y"], 0.5 * curve["line_x"], atol=1e-9)


def test_cal_curve_pht_two_step_chain():
    """par_pht: fits under the per-run step, the partition step applied on top."""
    det = synthetic_pars()["V01"]
    ops = det["pars"]["operations"]
    ops["cuspEmax_ctc_runcal"] = ops.pop("cuspEmax_ctc_cal")
    ops["cuspEmax_ctc_cal"] = {
        "expression": "a + b * cuspEmax_ctc_runcal",
        "parameters": {"a": 1.0, "b": 1.0},  # partition cal shifts by 1 keV
    }
    ecal = det["results"]["ecal"]
    ecal["cuspEmax_ctc_runcal"] = ecal.pop("cuspEmax_ctc_cal")

    curve = cal_curve({"V01": det}, "V01")

    np.testing.assert_allclose(curve["residual"], 1.0, atol=1e-9)
    assert "cuspEmax_ctc_runcal = a + b * cuspEmax_ctc" in curve["expression"]


def test_cal_curve_no_valid_peaks():
    pars = synthetic_pars()
    for fit in pars["V01"]["results"]["ecal"]["cuspEmax_ctc_cal"]["pk_fits"].values():
        fit["validity"] = False
    with pytest.raises(KeyError):
        cal_curve(pars, "V01")


def test_cal_residuals_order_and_nan():
    pars = {**synthetic_pars(), "A99": synthetic_pars()["V01"]}
    names, residuals = cal_residuals(pars, ["V01", "A99", "NOPE"])
    assert names == ["V01", "A99"]
    res, _err = residuals[2614.511]
    np.testing.assert_allclose(res, 0.0, atol=1e-9)
    res583, _ = residuals[583.191]
    assert np.isnan(res583).all()  # only invalid fits for that peak


def bulky_pars():
    """synthetic_pars plus the sections a real par file carries but we skip."""
    pars = synthetic_pars()
    det = pars["V01"]
    det["pars"]["dsp_config"] = {"filters": ["cusp", "zac"], "tau": 400.0}
    det["results"]["aoe"] = {"default": {"cut": [1.0, 2.0], "pk_fits": {"1": 2}}}
    det["results"]["ecal"]["zacEmax_ctc_cal"] = {"pk_fits": {"2614.511": {"a": 1}}}
    det["results"]["ecal"]["cuspEmax_ctc_cal"]["eres_linear"] = {"pars": [0.1, 0.2]}
    pars["V02"] = {"pars": {"operations": {}}}  # an off detector: no results
    return pars


def test_slim_cal_yaml_keeps_what_the_plots_read(tmp_path):
    f = tmp_path / "par_hit.yaml"
    f.write_text(yaml.safe_dump(bulky_pars(), sort_keys=False))

    slim = _read_cal_yaml(f)

    full = yaml.safe_load(f.read_text())
    assert slim["V01"]["pars"]["operations"] == full["V01"]["pars"]["operations"]
    assert slim["V01"]["results"]["ecal"] == {
        "cuspEmax_ctc_cal": full["V01"]["results"]["ecal"]["cuspEmax_ctc_cal"]
    }
    assert "aoe" not in slim["V01"]["results"]
    assert "dsp_config" not in slim["V01"]["pars"]
    assert slim["V02"] == full["V02"]
    np.testing.assert_allclose(
        cal_curve(slim, "V01")["residual"], cal_curve(full, "V01")["residual"]
    )


def test_read_cal_yaml_falls_back_on_an_unexpected_layout(tmp_path):
    f = tmp_path / "par_hit.yaml"
    f.write_text(yaml.safe_dump(synthetic_pars(), indent=4))  # not the dump layout

    pars = _read_cal_yaml(f)

    assert pars["V01"]["results"]["ecal"]["cuspEmax_ctc_cal"]["pk_fits"]
    assert cal_curve(pars, "V01")["peaks"].size == 2


def write_par_tier(root, tier, files, *, category=None, pars=None):
    """A fake ``par_<tier>`` tree: ``files`` = {(period, run, tstamp): pars}."""
    lines = []
    for (period, run, tstamp), det_pars in files.items():
        rel = f"cal/{period}/{run}/l200-{period}-{run}-cal-{tstamp}-par_{tier}.yaml"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(yaml.safe_dump(det_pars or pars))
        lines += [f"- valid_from: {tstamp}", "  mode: reset"]
        if category:
            lines.append(f"  category: {category}")
        lines += ["  apply:", f"  - {rel}"]
    (root / "validity.yaml").write_text("\n".join(lines) + "\n")


def cal_viewer(paths, tier="evt", start_key="20250601T000000Z"):
    viewer = FakeViewer(
        runs={"p02": {"r002": ["x"]}},
        paths={k: str(v) for k, v in paths.items()},
        runinfo={"p02": {"r002": {"phy": {"start_key": start_key}}}},
    )
    viewer.tier = tier
    return viewer


def test_load_cal_pars_through_validity(tmp_path):
    hit = tmp_path / "hit"
    write_par_tier(
        hit,
        "hit",
        {
            ("p01", "r001", "20250101T000000Z"): synthetic_pars(0.5),
            ("p02", "r001", "20250701T000000Z"): synthetic_pars(0.7),
        },
    )
    pars, label = ValidationData(cal_viewer({"par_hit": hit})).load_cal_pars(
        "p02", "r002"
    )
    # the entry valid at the run's start key, not the newest file on disk
    assert label == "cal pars (hit): p01 r001"
    assert pars["V01"]["pars"]["operations"]["cuspEmax_ctc_cal"]["parameters"][
        "b"
    ] == pytest.approx(0.5)


def test_load_cal_pars_cal_only_catalog(tmp_path):
    # a catalog listing its entries under ``category: cal`` only: querying the
    # default ``all`` finds nothing, which used to fail every run
    hit = tmp_path / "hit"
    write_par_tier(
        hit,
        "hit",
        {("p01", "r001", "20250101T000000Z"): None},
        category="cal",
        pars=synthetic_pars(),
    )
    pars, label = ValidationData(cal_viewer({"par_hit": hit})).load_cal_pars(
        "p02", "r002"
    )
    assert pars is not None
    assert label == "cal pars (hit): p01 r001"


@pytest.mark.parametrize(("tier", "first"), [("pet", "pht"), ("evt", "hit")])
def test_cal_par_tier_preference(tmp_path, tier, first):
    paths = {}
    for t in ("hit", "pht"):
        paths[f"par_{t}"] = tmp_path / t
        write_par_tier(
            tmp_path / t, t, {("p01", "r001", "20250101T000000Z"): synthetic_pars()}
        )
    sources, _ = ValidationData(cal_viewer(paths, tier=tier)).cal_par_sources(
        "p02", "r002"
    )
    assert [s[0] for s in sources] == [first, "pht" if first == "hit" else "hit"]


def test_pht_only_cycle(tmp_path):
    pht = tmp_path / "pht"
    write_par_tier(pht, "pht", {("p01", "r001", "20250101T000000Z"): synthetic_pars()})
    viewer = cal_viewer({"par_hit": tmp_path / "nohit", "par_pht": pht})
    pars, label = ValidationData(viewer).load_cal_pars("p02", "r002")
    assert pars is not None
    assert label == "cal pars (pht): p01 r001"


def test_load_cal_pars_nothing_found_says_where(tmp_path):
    pht = tmp_path / "pht"
    pht.mkdir()
    # listed in validity.yaml but never written (as in mock_prod's par/pht)
    (pht / "validity.yaml").write_text(
        "- valid_from: 20250101T000000Z\n  apply:\n"
        "  - cal/p01/r001/l200-p01-r001-cal-20250101T000000Z-par_pht.yaml\n"
    )
    viewer = cal_viewer({"par_hit": tmp_path / "nowhere", "par_pht": pht})
    pars, reason = ValidationData(viewer).load_cal_pars("p02", "r002")
    assert pars is None
    assert "20250601T000000Z" in reason
    assert f"par_hit={tmp_path / 'nowhere'} (no validity.yaml)" in reason
    assert "listed file missing" in reason

    # before the first validity entry
    hit = tmp_path / "hit"
    write_par_tier(hit, "hit", {("p01", "r001", "20250101T000000Z"): synthetic_pars()})
    viewer = cal_viewer({"par_hit": hit}, start_key="20240101T000000Z")
    pars, reason = ValidationData(viewer).load_cal_pars("p02", "r002")
    assert pars is None
    assert "no par_hit entry valid" in reason

    # cycle without any par path at all
    pars, reason = ValidationData(cal_viewer({})).load_cal_pars("p02", "r002")
    assert pars is None
    assert "no par_hit/par_pht path" in reason


def test_view_builders_smoke():
    """The figure builders accept real period_series output."""
    t = T0 + np.arange(0, 7200, 10.0)
    k40 = K_LINES["K40"]
    data = make_data(
        {"r001": columns(t, energy=ak.Array([[k40]] * t.size))},
    )
    times, rates = data.period_series("p01", 3600)
    for name, builder in validation_view.RATE_BUILDERS.items():
        for log_y in (True, False):
            assert builder(times, rates, "1 h", log_y=log_y) is not None, name

    names, residuals = cal_residuals(synthetic_pars(), ["V01"])
    assert validation_view.cal_summary(names, residuals, [1], "cal pars: test")
    curve = cal_curve(synthetic_pars(), "V01")
    assert validation_view.cal_detail(curve, "V01", "cal pars: test")
