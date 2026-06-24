from __future__ import annotations

import math
import pathlib

import lgdo.lh5_store as lh5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dspeed.vis import WaveformBrowser
from PyQt5.QtCore import QObject, pyqtSignal

mpl.use("Qt5Agg")


def plot_string_compressed(fig, ax, ev, browsers, line_parameter="wf_blsub", string=1):
    def onpick(event, lined):
        legline = event.artist
        if event.mouseevent.dblclick:
            for entry, line in lined.items():
                if entry is not legline:
                    entry.set_alpha(0.2)
                    line.set_visible(False)
                else:
                    entry.set_alpha(1)
                    line.set_visible(True)
        else:
            line = lined[legline]
            # If the label of the legend matches that of the line, then set the alpha to 0
            # Change the alpha on the line in the legend, so we can see what lines
            # have been toggled.
            visible = not line.get_visible()
            legline.set_alpha(1.0 if visible else 0.2)
            line.set_visible(visible)
        fig.canvas.draw()

    dets = [
        det_dict["name"]
        for pos, det_dict in ev.chmap.map("location.string", unique=False)[string]
        .map("location.position")
        .items()
        if det_dict["analysis"]["processable"] is True
    ]

    det_lines = {}
    lines = []
    for i, det in enumerate(dets):
        browser = browsers[f"ch{ev.chmap[det].daq.rawid}"]
        line = browser.lines[line_parameter][0]
        x_lim = browser.x_lim if browser.x_lim else browser.auto_x_lim
        y_lim = browser.y_lim
        if not y_lim:
            y_range = browser.auto_y_lim[1] - browser.auto_y_lim[0]
            y_lim = [
                browser.auto_y_lim[0] - 0.05 * y_range,
                browser.auto_y_lim[1] + 0.05 * y_range,
            ]
        browser.ax.set_xlim(*x_lim)
        browser.ax.set_ylim(*y_lim)

        line.set_transform(browser.ax.transData)
        det_lines[det] = line
        line.set_color(mpl.colormaps["tab20"](i / len(dets)))
        lines.append(line)
        ax.add_line(line)
        ax.set_xlabel(browser.x_unit)
        if browser.x_lim:
            ax.set_xlim(*browser.x_lim)
        plt.plot([], color=mpl.colormaps["tab20"](i / len(dets)), label=det, lw=5)

    leg = ax.legend(ncol=2, loc="upper left", frameon=False)

    lined = {}
    for legline in leg.get_lines():
        legline.set_picker(True)
        lined[legline] = det_lines[legline.get_label()]

    low, high = ax.yaxis.get_data_interval()
    if low > -200:
        low = -200
    if high < 200:
        high = 200
    ax.set_ylim([low, high])
    ax.set_ylabel("value (adc)")
    ax.set_xlabel("time (ns)")
    fig.canvas.mpl_connect("pick_event", lambda x: onpick(x, lined))
    # fig.canvas.draw()


def plot_above_threshold_compressed(fig, ax, ev, browsers, line_parameter="wf_blsub"):
    def onpick(event, lined):
        legline = event.artist
        if event.mouseevent.dblclick:
            for entry, line in lined.items():
                if entry is not legline:
                    entry.set_alpha(0.2)
                    line.set_visible(False)
                else:
                    entry.set_alpha(1)
                    line.set_visible(True)
        else:
            line = lined[legline]
            # If the label of the legend matches that of the line, then set the alpha to 0
            # Change the alpha on the line in the legend, so we can see what lines
            # have been toggled.
            visible = not line.get_visible()
            legline.set_alpha(1.0 if visible else 0.2)
            line.set_visible(visible)
        fig.canvas.draw()

    dets = [
        det
        for det, energy in ev.energy_dict.items()
        if energy is not None and (energy > 25)
    ]

    det_lines = {}
    lines = []
    for i, det in enumerate(dets):
        browser = browsers[f"ch{ev.chmap[det].daq.rawid}"]
        line = browser.lines[line_parameter][0]
        x_lim = browser.x_lim if browser.x_lim else browser.auto_x_lim
        y_lim = browser.y_lim
        if not y_lim:
            y_range = browser.auto_y_lim[1] - browser.auto_y_lim[0]
            y_lim = [
                browser.auto_y_lim[0] - 0.05 * y_range,
                browser.auto_y_lim[1] + 0.05 * y_range,
            ]
        browser.ax.set_xlim(*x_lim)
        browser.ax.set_ylim(*y_lim)

        line.set_transform(browser.ax.transData)
        det_lines[det] = line
        line.set_color(mpl.colormaps["tab20"](i / len(dets)))
        lines.append(line)
        ax.add_line(line)
        ax.set_xlabel(browser.x_unit)
        if browser.x_lim:
            ax.set_xlim(*browser.x_lim)
        if len(dets) < 10:
            plt.plot([], color=mpl.colormaps["tab20"](i / len(dets)), label=det, lw=5)

    if len(dets) < 10:
        leg = ax.legend(ncol=2, loc="upper left", frameon=False)

        lined = {}
        for legline in leg.get_lines():
            legline.set_picker(True)
            lined[legline] = det_lines[legline.get_label()]

    low, high = ax.yaxis.get_data_interval()
    if low > -200:
        low = -200
    if high < 200:
        high = 200
    ax.set_ylim([low, high])
    ax.set_ylabel("value (adc)")
    ax.set_xlabel("time (ns)")
    if len(dets) < 10:
        fig.canvas.mpl_connect("pick_event", lambda x: onpick(x, lined))


def plot_all_compressed(fig, ax, ev, browsers, line_parameter="wf_blsub"):
    def onpick(event, lined):
        legline = event.artist
        if event.mouseevent.dblclick:
            for entry, lines in lined.items():
                if entry is not legline:
                    entry.set_alpha(0.2)
                    for line in lines:
                        line.set_visible(False)
                else:
                    entry.set_alpha(1)
                    for line in lines:
                        line.set_visible(True)
        else:
            lines = lined[legline]
            # If the label of the legend matches that of the line, then set the alpha to 0
            # Change the alpha on the line in the legend, so we can see what lines
            # have been toggled.
            visible = not lines[0].get_visible()
            legline.set_alpha(1.0 if visible else 0.2)
            for line in lines:
                line.set_visible(visible)
        fig.canvas.draw()

    strings = {}
    lines = []
    for chan, browser in browsers.items():
        det = ev.chmap.map("daq.rawid")[int(chan[2:])]["name"]
        line = browser.lines[line_parameter][0]
        x_lim = browser.x_lim if browser.x_lim else browser.auto_x_lim
        y_lim = browser.y_lim
        if not y_lim:
            y_range = browser.auto_y_lim[1] - browser.auto_y_lim[0]
            y_lim = [
                browser.auto_y_lim[0] - 0.05 * y_range,
                browser.auto_y_lim[1] + 0.05 * y_range,
            ]
        browser.ax.set_xlim(*x_lim)
        browser.ax.set_ylim(*y_lim)

        line.set_transform(browser.ax.transData)
        string = ev.chmap[det]["location"]["string"]
        if string in list(strings):
            strings[string].append(line)
        else:
            strings[string] = [line]
        line.set_color(mpl.colormaps["tab20"](string / 12))
        lines.append(line)
        ax.add_line(line)
        ax.set_xlabel(browser.x_unit)
        if browser.x_lim:
            ax.set_xlim(*browser.x_lim)

    for string in np.unique(list(strings)):
        plt.plot(
            [],
            color=mpl.colormaps["tab20"](string / 12),
            label=f"String:{string:02}",
            lw=5,
        )

    leg = ax.legend(ncol=1, loc="upper left", frameon=False)  #

    lined = {}
    for legline in leg.get_lines():
        legline.set_picker(True)
        lined[legline] = strings[int(legline.get_label()[-2:])]

    low, high = ax.yaxis.get_data_interval()
    if low > -200:
        low = -200
    if high < 200:
        high = 200
    ax.set_ylim([low, high])
    ax.set_ylabel("value (adc)")
    ax.set_xlabel("time (ns)")
    fig.canvas.mpl_connect("pick_event", lambda x: onpick(x, lined))


def plot_event_compressed(
    ev, fig, browsers, line_parameter="wf_blsub", plot_category="all"
):
    ax = fig.add_subplot(111)
    for _det, browser in browsers.items():
        browser.set_figure(fig, ax)
        browser.find_entry(ev.index, append=False)
    if plot_category == "all":
        plot_all_compressed(fig, ax, ev, browsers, line_parameter)
    elif plot_category == "above threshold":
        plot_above_threshold_compressed(fig, ax, ev, browsers, line_parameter)
    else:
        plot_string_compressed(
            fig, ax, ev, browsers, line_parameter, string=int(plot_category[-2:])
        )


def plot_string_exploded(fig, ev, browsers, line_parameter="wf_blsub", string=1):
    dets = [
        det_dict["name"]
        for pos, det_dict in ev.chmap.map("location.string", unique=False)[string]
        .map("location.position")
        .items()
        if det_dict["analysis"]["processable"] is True
    ]

    if len(dets) < 6:
        det_ax = {dets[0]: plt.subplot(math.ceil(len(dets) / 2), 2, 1)}
        for i, det in enumerate(dets[1:]):
            det_ax[det] = plt.subplot(
                math.ceil(len(dets) / 2),
                2,
                i + 2,
                sharex=det_ax[dets[0]],
                sharey=det_ax[dets[0]],
            )
    else:
        det_ax = {dets[0]: plt.subplot(math.ceil(len(dets) / 3), 3, 1)}
        for i, det in enumerate(dets[1:]):
            det_ax[det] = plt.subplot(
                math.ceil(len(dets) / 3),
                3,
                i + 2,
                sharex=det_ax[dets[0]],
                sharey=det_ax[dets[0]],
            )

    low = -200
    high = 200

    for i, det in enumerate(dets):
        browser = browsers[f"ch{ev.chmap[det].daq.rawid}"]
        ax = det_ax[det]
        browser.set_figure(fig, ax)

        line = browser.lines[line_parameter][0]
        x_lim = browser.x_lim if browser.x_lim else browser.auto_x_lim
        y_lim = browser.y_lim
        if not y_lim:
            y_range = browser.auto_y_lim[1] - browser.auto_y_lim[0]
            y_lim = [
                browser.auto_y_lim[0] - 0.05 * y_range,
                browser.auto_y_lim[1] + 0.05 * y_range,
            ]
        browser.ax.set_xlim(*x_lim)
        browser.ax.set_ylim(*y_lim)

        line.set_transform(browser.ax.transData)

        line.set_color(mpl.colormaps["tab20"](i / len(dets)))
        ax.add_line(line)
        ax.set_xlabel(browser.x_unit)
        if browser.x_lim:
            ax.set_xlim(*browser.x_lim)

        current_low, current_high = ax.yaxis.get_data_interval()
        if current_low < low:
            low = current_low
        if current_high > high:
            high = current_high

    for det in list(dets):
        ax = det_ax[det]
        ax.scatter([], [], color="white", label=det, marker="x", s=0.0001)
        ax.legend(ncol=1, loc="upper left", frameon=False, markerfirst=False)
        ax.set_ylim([low, high])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.label_outer()
    fig.text(0.5, 0.02, "time (ns)", ha="center", va="center")
    fig.text(0.015, 0.5, "value (adu)", ha="center", va="center", rotation="vertical")
    fig.tight_layout()


def plot_above_threshold_exploded(fig, ev, browsers, line_parameter="wf_blsub"):
    dets = [
        det
        for det, energy in ev.energy_dict.items()
        if energy is not None and (energy > 25)
    ]
    if len(dets) > 12:
        dets = dets[:12]
    if len(dets) == 1:
        det_ax = {dets[0]: plt.subplot(111)}

    elif len(dets) < 6:
        det_ax = {dets[0]: plt.subplot(math.ceil(len(dets) / 2), 2, 1)}
        for i, det in enumerate(dets[1:]):
            det_ax[det] = plt.subplot(
                math.ceil(len(dets) / 2),
                2,
                i + 2,
                sharex=det_ax[dets[0]],
                sharey=det_ax[dets[0]],
            )
    else:
        det_ax = {dets[0]: plt.subplot(math.ceil(len(dets) / 3), 3, 1)}
        for i, det in enumerate(dets[1:]):
            det_ax[det] = plt.subplot(
                math.ceil(len(dets) / 3),
                3,
                i + 2,
                sharex=det_ax[dets[0]],
                sharey=det_ax[dets[0]],
            )

    low = -200
    high = 200

    for i, det in enumerate(dets):
        browser = browsers[f"ch{ev.chmap[det].daq.rawid}"]
        ax = det_ax[det]
        browser.set_figure(fig, ax)

        line = browser.lines[line_parameter][0]
        x_lim = browser.x_lim if browser.x_lim else browser.auto_x_lim
        y_lim = browser.y_lim
        if not y_lim:
            y_range = browser.auto_y_lim[1] - browser.auto_y_lim[0]
            y_lim = [
                browser.auto_y_lim[0] - 0.05 * y_range,
                browser.auto_y_lim[1] + 0.05 * y_range,
            ]
        browser.ax.set_xlim(*x_lim)
        browser.ax.set_ylim(*y_lim)

        line.set_transform(browser.ax.transData)

        line.set_color(mpl.colormaps["tab20"](i / len(dets)))
        ax.add_line(line)
        ax.set_xlabel(browser.x_unit)
        if browser.x_lim:
            ax.set_xlim(*browser.x_lim)

        current_low, current_high = ax.yaxis.get_data_interval()
        if current_low < low:
            low = current_low
        if current_high > high:
            high = current_high

    for det in list(dets):
        ax = det_ax[det]
        ax.scatter([], [], color="white", label=det, marker="x", s=0.0001)
        ax.legend(ncol=1, loc="upper left", frameon=False, markerfirst=False)
        ax.set_ylim([low, high])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.label_outer()
    fig.text(0.5, 0.02, "time (ns)", ha="center", va="center")
    fig.text(0.015, 0.5, "value (adu)", ha="center", va="center", rotation="vertical")
    fig.tight_layout()


def plot_all_exploded(fig, ev, browsers, line_parameter="wf_blsub"):
    string_list = np.unique(list(ev.chmap.map("location.string", unique=False)))
    strings = {sorted(string_list)[0]: plt.subplot(4, 3, 1)}
    for i, string in enumerate(sorted(string_list)[1:]):
        strings[string] = plt.subplot(
            4,
            3,
            i + 2,
            sharex=strings[sorted(string_list)[0]],
            sharey=strings[sorted(string_list)[0]],
        )

    low = -200
    high = 200

    lines = []
    for chan, browser in browsers.items():
        det = ev.chmap.map("daq.rawid")[int(chan[2:])]["name"]
        string = ev.chmap[det]["location"]["string"]
        ax = strings[string]
        browser.set_figure(fig, ax)

        line = browser.lines[line_parameter][0]
        x_lim = browser.x_lim if browser.x_lim else browser.auto_x_lim
        y_lim = browser.y_lim
        if not y_lim:
            y_range = browser.auto_y_lim[1] - browser.auto_y_lim[0]
            y_lim = [
                browser.auto_y_lim[0] - 0.05 * y_range,
                browser.auto_y_lim[1] + 0.05 * y_range,
            ]
        browser.ax.set_xlim(*x_lim)
        browser.ax.set_ylim(*y_lim)

        line.set_transform(browser.ax.transData)

        line.set_color(mpl.colormaps["tab20"](string / 12))
        lines.append(line)
        ax.add_line(line)
        ax.set_xlabel(browser.x_unit)
        if browser.x_lim:
            ax.set_xlim(*browser.x_lim)

        current_low, current_high = ax.yaxis.get_data_interval()
        if current_low < low:
            low = current_low
        if current_high > high:
            high = current_high

    for string in list(strings):
        ax = strings[string]
        ax.scatter(
            [], [], color="white", label=f"String:{string:02}", marker="x", s=0.0001
        )
        ax.legend(ncol=1, loc="upper left", frameon=False, markerfirst=False)  #

        ax.set_ylim([low, high])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.label_outer()
    fig.text(0.5, 0.02, "time (ns)", ha="center", va="center")
    fig.text(0.015, 0.5, "value (adu)", ha="center", va="center", rotation="vertical")
    fig.tight_layout()


def plot_event_exploded(
    ev, fig, browsers, line_parameter="wf_blsub", plot_category="all"
):
    for _det, browser in browsers.items():
        browser.find_entry(ev.index, append=False)
    if plot_category == "all":
        plot_all_exploded(fig, ev, browsers, line_parameter)
    elif plot_category == "above threshold":
        plot_above_threshold_exploded(fig, ev, browsers, line_parameter)
    else:
        plot_string_exploded(
            fig, ev, browsers, line_parameter, string=int(plot_category[-2:])
        )


class spoof_iterator:
    sto = lh5.LH5Store()

    def __init__(self, ev, channel):
        self.ev = ev
        self.channel = channel
        f = self.ev.raw_file
        g = f"{self.channel}/raw"
        self.lh5_buffer = self.sto.get_buffer(
            g,
            f,
            size=1,
        )
        self.current_entry = np.nan
        self.n_rows = 1

    def read(self, entry):  # noqa: ARG002
        self.lh5_buffer, n_rows = self.sto.read_object(
            f"{self.channel}/raw", self.ev.raw_file, start_row=self.ev.index, n_rows=1
        )
        return (self.lh5_buffer, self.n_rows)


class event_waveform_browser:
    def __init__(self, ev):
        self.cached_browsers = {}
        self.ev = ev
        self.ev._get_event("*", "*", "*", 0)
        self.file = self.ev.raw_file
        self.browsers = {}
        self.current_dsp_configs = {}

    def _get_browser(self, channel, dsp_config):
        browser = WaveformBrowser(
            self.ev.raw_file,
            f"{channel}/raw",
            dsp_config=dsp_config,
            lines=["wf_blsub", "wf_pz", "wf_trap", "curr"],  # change this to use config
            buffer_len=1,
        )
        spoof = spoof_iterator(self.ev, channel)
        browser.lh5_it = spoof
        # need to replace browsers lh5_its
        return browser

    def get_browsers(self):
        channels = [f"ch{self.ev.chmap[det].daq.rawid}" for det in self.ev.working_dets]
        dataprod_config = self.ev.meta.dataprod.config.on(
            str(pathlib.Path(self.file).name).split("-")[4]
        )
        dsp_configs = dataprod_config["snakemake_rules"]["tier_dsp"]["inputs"][
            "processing_chain"
        ]
        # check if channels are same
        if channels == list(self.browsers):
            # check dsp configs are same
            for channel in channels:
                if dsp_configs[f"{channel}/raw"] != self.current_dsp_configs[channel]:
                    if channel in list(self.cached_browsers):
                        self.cached_browsers[channel].update(
                            {
                                self.current_dsp_configs[channel]: self.browsers.pop(
                                    channel
                                )
                            }
                        )
                        if dsp_configs[channel] in list(self.cached_browsers[channel]):
                            self.browsers[channel] = self.cached_browsers[channel].pop(
                                dsp_configs[f"{channel}/raw"]
                            )
                        else:
                            self.browsers[channel] = self._get_browser(
                                channel, dsp_configs[f"{channel}/raw"]
                            )
                    else:
                        self.cached_browsers[channel] = {
                            self.current_dsp_configs[channel]: self.browsers.pop(
                                channel
                            )
                        }
                        self.browsers[channel] = self._get_browser(
                            channel, dsp_configs[f"{channel}/raw"]
                        )

        else:
            # check for channels missing
            for channel in channels:
                if channel not in list(self.browsers):
                    if channel in list(self.cached_browsers):
                        if dsp_configs[f"{channel}/raw"] in list(
                            self.cached_browsers[channel]
                        ):
                            self.browsers[channel] = self.cached_browsers[channel][
                                dsp_configs[f"{channel}/raw"]
                            ].pop()
                        else:
                            self.browsers[channel] = self._get_browser(
                                channel, dsp_configs[f"{channel}/raw"]
                            )
                    else:
                        self.browsers[channel] = self._get_browser(
                            channel, dsp_configs[f"{channel}/raw"]
                        )
            # check for extra channels
            for channel, _browser in self.browsers.items():
                if channel not in channels:
                    if channel in list(self.cached_browsers):
                        self.cached_browsers[channel].update(
                            {
                                self.current_dsp_configs[channel]: self.browsers.pop(
                                    channel
                                )
                            }
                        )
                    else:
                        self.cached_browsers[channel] = {
                            self.current_dsp_configs[channel]: self.browsers.pop(
                                channel
                            )
                        }
        self.current_dsp_configs = {
            name.split("/")[0]: item for name, item in dsp_configs.items()
        }


class getBrowsers(QObject):
    finished = pyqtSignal()

    def __init__(self, browsers):
        self.browsers = browsers
        super().__init__()

    def run(self):
        self.browsers.get_browsers()
        self.finished.emit()
