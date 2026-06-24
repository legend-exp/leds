"""geds energy spectra from the evt tier, with optional cuts.

Used by two views: the event-display tab's *accumulating* raw spectrum
(``upto_index`` set, no cuts) and the Spectrum tab's *whole-run* spectrum
(1 keV bins, cut toggles). The run's per-hit ``evt/geds/energy`` and the cut
arrays are read once and cached, so re-histogramming on a cut toggle is cheap.
"""

from __future__ import annotations

import awkward as ak
import lh5
import numpy as np

#: Binary cuts: key -> (label, positive-option label, negative-option label).
#: Each option is an independent checkbox; the positive option keeps events
#: where the cut condition holds, the negative keeps where it does not. With
#: neither or both ticked the cut is off. The condition per key is in
#: ``_event_mask`` / ``_kept_energy``.
BINARY_CUTS = {
    "geds_trigger": ("geds trigger", "forced", "normal"),
    "muon": ("muon", "coincident", "anticoincident"),
    "spms": ("spms", "coincident", "anticoincident"),
    "quality": ("quality", "pass", "fail"),
    "psd": ("psd", "pass", "fail"),
}
MULT_OPTIONS = ("off", "1", "2", ">2")

#: Energy axis of the spectra.
ENERGY_RANGE = (0, 4000)
N_BINS = 1000  # ~4 keV bins (accumulating playback spectrum)
DEFAULT_BIN_WIDTH = 5.0  # keV (Spectrum tab, user-adjustable)


def bins_for_width(width):
    """Number of histogram bins over ``ENERGY_RANGE`` for a ``width`` keV bin."""
    return max(1, round((ENERGY_RANGE[1] - ENERGY_RANGE[0]) / width))


class RunSpectrum:
    """Cut-and-histogram the geds energies of a run."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._cache: dict = {}

    def _load(self, period, run):
        key = (period, run)
        if key not in self._cache:
            files = [str(f) for f in self.viewer._run_files(period, run)]
            group = self.viewer.group  # in-file group ("evt"), not the file tier

            def col(field):
                return lh5.read(f"{group}/{field}", files)

            self._cache[key] = {
                "energy": col("geds/energy").view_as("ak"),
                "psd_bb": ak.values_astype(
                    col("geds/psd/is_bb_like").view_as("ak"), bool
                ),
                "puls": col("coincident/puls").nda.astype(bool),
                "muon": col("coincident/muon").nda.astype(bool),
                "spms": col("coincident/spms").nda.astype(bool),
                "forced": col("trigger/is_forced").nda.astype(bool),
                "qc": col("geds/quality/is_bb_like").nda.astype(bool),
                "mult": col("geds/multiplicity").nda,
            }
        return self._cache[key]

    @staticmethod
    def _apply_binary(keep, cond, checked):
        """Tighten ``keep`` by a binary cut's (positive, negative) checkboxes."""
        pos, neg = checked
        if pos and not neg:
            keep &= cond
        elif neg and not pos:
            keep &= ~cond
        return keep  # neither or both ticked -> no filter

    @staticmethod
    def _event_mask(d, cuts):
        """Per-event boolean mask for the event-level cuts.

        Each binary cut keeps events where its condition holds (positive option:
        geds_trigger=is_forced, muon/spms=their coincidence, quality=bb-like) or
        does not (negative option). Multiplicity selects 1 / 2 / >2.
        """
        keep = np.ones(len(d["puls"]), dtype=bool)
        conditions = {
            # forced = is_forced | pulser-coincident; normal = its complement
            "geds_trigger": d["forced"] | d["puls"],
            "muon": d["muon"],
            "spms": d["spms"],
            "quality": d["qc"],
        }
        for key, cond in conditions.items():
            keep = RunSpectrum._apply_binary(keep, cond, cuts.get(key, (False, False)))

        mult = cuts.get("multiplicity", "off")
        if mult == "1":
            keep &= d["mult"] == 1
        elif mult == "2":
            keep &= d["mult"] == 2
        elif mult == ">2":
            keep &= d["mult"] > 2
        return keep

    @staticmethod
    def _kept_energy(d, keep, cuts):
        """Per-hit energies of the kept events, with the per-hit psd cut applied."""
        energy = d["energy"][keep]
        pos, neg = cuts.get("psd", (False, False))
        if pos and not neg:
            energy = energy[d["psd_bb"][keep]]
        elif neg and not pos:
            energy = energy[~d["psd_bb"][keep]]
        return energy

    def histogram(self, period, run, *, upto_index=None, cuts=None, bins=N_BINS):
        """Return ``(counts, edges)`` of the geds energies passing the cuts.

        ``upto_index`` (if given) restricts to events ``0..upto_index``;
        ``cuts`` is a ``{cut_key: bool}`` mapping (see ``CUTS``).
        """
        d = self._load(period, run)
        cuts = cuts or {}
        keep = self._event_mask(d, cuts)
        if upto_index is not None:
            keep[max(0, min(int(upto_index), len(keep) - 1)) + 1 :] = False
        values = ak.to_numpy(ak.flatten(self._kept_energy(d, keep, cuts)))
        return np.histogram(values, bins=bins, range=ENERGY_RANGE)

    def events_in_bin(self, period, run, cuts, lo, hi):
        """Run-global indices of events with a passing geds hit in ``[lo, hi)``."""
        d = self._load(period, run)
        cuts = cuts or {}
        keep = self._event_mask(d, cuts)
        energy = self._kept_energy(d, keep, cuts)
        in_bin = ak.to_numpy(ak.any((energy >= lo) & (energy < hi), axis=1))
        return np.flatnonzero(keep)[in_bin]
