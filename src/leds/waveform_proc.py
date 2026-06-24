"""DSP-processed waveforms via dspeed's :class:`WaveformBrowser`.

The browser already builds the per-detector processing chain (resolving the
``db.*`` parameters from the dataprod config) and exposes the intermediate
waveforms as matplotlib lines; we reuse it purely as a processing engine and
pull the ``(x, y)`` data out to draw in Bokeh.

Browsers are expensive to build, so one is cached per ``(raw_file, channel)``
and reused across events via :meth:`WaveformBrowser.find_entry`.
"""

from __future__ import annotations

import matplotlib as mpl

# WaveformBrowser draws onto matplotlib axes internally; keep it headless.
mpl.use("Agg")

from dspeed.vis import WaveformBrowser

#: Intermediate processing-chain waveforms offered by the dsp view.
PROCESSED_PARAMS = ("wf_blsub", "wf_pz", "wf_trap", "curr")


class WaveformProcessor:
    """Lazily build/cache dspeed browsers and return processed traces."""

    def __init__(self, ev):
        self.ev = ev
        self._browsers: dict = {}  # (raw_file, channel) -> WaveformBrowser
        self._dsp_cfgs: dict = {}  # tstamp -> {detector_name: dsp_config_path}

    def _dsp_config(self, name):
        ts = self.ev.tstamp
        if ts not in self._dsp_cfgs:
            cfg = self.ev.meta.dataprod.config.on(ts)
            self._dsp_cfgs[ts] = cfg["snakemake_rules"]["tier_dsp"]["inputs"][
                "processing_chain"
            ]
        return self._dsp_cfgs[ts][name]

    def _browser(self, rawid, name):
        raw_file = str(self.ev.raw_file())
        channel = f"ch{rawid:07d}"
        key = (raw_file, channel)
        if key not in self._browsers:
            self._browsers[key] = WaveformBrowser(
                raw_file,
                f"{channel}/raw",
                dsp_config=self._dsp_config(name),
                lines=list(PROCESSED_PARAMS),
                buffer_len=1,
            )
        return self._browsers[key]

    def processed(self, rawid, name, hit_idx, param):
        """Return ``(x, y)`` arrays for ``param`` at this detector's hit row."""
        browser = self._browser(rawid, name)
        browser.find_entry(hit_idx, append=False)
        line = browser.lines[param][0]
        return line.get_xdata(), line.get_ydata()
