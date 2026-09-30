from __future__ import annotations

import numpy as np
import pytest

from leds import calcheck as C


def th_spectrum(scale=1.0, n=60_000, seed=0):
    """A Th-228-like spectrum on the stored bins: lines on a falling continuum."""
    rng = np.random.default_rng(seed)
    e = [rng.exponential(600.0, n)]
    for line, frac in zip(C.TH_LINES, (0.2, 0.08, 0.02, 0.02, 0.01, 0.015, 0.05), strict=True):
        e.append(rng.normal(line, 1.2, int(frac * n)))
    h, _ = np.histogram(np.concatenate(e) * scale, bins=C.N_BINS, range=(0, C.E_MAX))
    return h.astype(float)


def test_scale_match_finds_gain_and_misidentified_peaks():
    good = [th_spectrum(seed=s) for s in range(5)]
    spectra = np.array([*good, th_spectrum(1.003, seed=9), th_spectrum(2614.511 / 2103.511, seed=8)])

    err, match = C.scale_match(spectra)

    np.testing.assert_allclose(err[:5], 0.0, atol=0.03)
    assert err[5] == pytest.approx(0.3, abs=0.03)
    assert err[6] == pytest.approx((2614.511 / 2103.511 - 1) * 100, abs=0.3)
    assert match[:6].min() > 0.8


def test_scale_match_leaves_thin_spectra_out():
    spectra = np.array([th_spectrum(), th_spectrum(n=500)])
    err, match = C.scale_match(spectra)
    assert np.isfinite(err[0])
    assert np.isnan(err[1])
    assert np.isnan(match[1])


def test_sections_and_their_evaluation():
    ops = {
        "is_valid_cal": {"expression": "is_valid_baseline & (is_valid_dteff_cal) & bl_pileup_cut"},
        "is_valid_baseline": {"expression": "is_valid_bl_slope & is_valid_bl_poly_rms"},
        "is_valid_dteff_cal": {"expression": "(dt_eff < a) | is_low_cuspEmax", "parameters": {"a": 5.0}},
    }
    cols = {
        "is_valid_bl_slope": np.array([1, 1, 0, 1], bool),
        "is_valid_bl_poly_rms": np.array([1, 0, 1, 1], bool),
        "dt_eff": np.array([1.0, 1.0, 1.0, 9.0]),
        "is_low_cuspEmax": np.array([0, 0, 0, 0], bool),
        "bl_pileup_cut": np.array([1, 1, 1, 1], bool),
    }
    sections = C.section_names(ops)
    assert sections == ["is_valid_baseline", "is_valid_dteff_cal", "bl_pileup_cut"]
    assert C.needed_fields("is_valid_dteff_cal", ops, set(cols)) == {"dt_eff", "is_low_cuspEmax"}
    np.testing.assert_array_equal(C.evaluate("is_valid_baseline", ops, cols), [1, 0, 0, 1])
    np.testing.assert_array_equal(C.evaluate("is_valid_dteff_cal", ops, cols), [1, 1, 1, 0])
    with pytest.raises(KeyError):
        C.needed_fields("is_valid_tail", ops, set(cols))


def test_cut_hist_matches_direct_masking():
    rng = np.random.default_rng(1)
    e = rng.uniform(0, C.E_MAX, 5000)
    masks = rng.integers(0, 8, 5000).astype(np.uint8)  # 3 sections
    full = 0b111
    det = {
        "sections": ["a", "b", "c"],
        "missing": set(),
        "pass": np.histogram(e[masks == full], C.N_BINS, (0, C.E_MAX))[0].astype(np.float32),
        "fail_e": e[masks != full],
        "fail_mask": masks[masks != full],
    }
    for selected, bits in ([], 0), (["a"], 0b001), (["a", "c"], 0b101), (["a", "b", "c"], full):
        direct, _ = np.histogram(e[(masks & bits) == bits], C.N_BINS, (0, C.E_MAX))
        mine, missing = C.cut_hist(det, selected)
        np.testing.assert_array_equal(mine, direct)
        assert missing == []
    assert C.cut_hist(det, ["a", "zzz"])[1] == ["zzz"]


def test_hist_by_rawid_and_pick_files():
    e = np.array([10.2, 10.4, 2999.9, 3000.0, np.nan, 5.0])
    r = np.array([1, 1, 2, 2, 1, 7])
    h = C.hist_by_rawid(e, r, [1, 2])
    assert h[1][20] == 2  # both in the 10.0-10.5 keV bin; NaN dropped
    assert h[2].sum() == 1  # 3000 keV is outside the range
    assert set(h) == {1, 2}  # rawid 7 is not a ged
    assert C.pick_files([f"f{i}" for i in range(10)], 3) == ["f0", "f4", "f9"]
    assert C.pick_files(["b", "a"], 3) == ["a", "b"]
