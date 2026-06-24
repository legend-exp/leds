# Archived legacy code

These files are the original **PyQt5** event display (and its helpers), kept for
reference only. They are **not** part of the installed `leds` package, are not
shipped, and are **not functional** against the current data/APIs (they predate
the move to the `evt` tier, the `dataflow-config.yaml` config format, and the
`lh5` IO package).

The live application is the Panel/Bokeh dashboard under `src/leds/`
(`leds serve` / `leds app`).

| file | what it was |
|------|-------------|
| `core.py` | PyQt5 `MainWindow` event display |
| `waveform_browse.py` | PyQt5 + dspeed waveform browser (matplotlib `pick_event`) |
| `layouts/*.ui` | Qt Designer UI files loaded via `uic` |
| `utils.py` | old `config.json`-based `sorter` (superseded by channelmap logic) |
