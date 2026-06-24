from __future__ import annotations

import contextlib
import os
import pathlib
import time

import boost_histogram as bh
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
)

from leds.event_viewer import event_viewer
from leds.waveform_browse import (
    event_waveform_browser,
    getBrowsers,
    plot_event_compressed,
    plot_event_exploded,
)

mpl.use("Qt5Agg")


class MplCanvas(FigureCanvasQTAgg):
    # first_call=True

    def __init__(self, width=6, height=5, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        # self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class ehist_window(QWidget):
    def __init__(self, ev):
        super().__init__()
        self.ev = ev
        self.canvas = MplCanvas(width=10, height=10, dpi=100)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ehist = bh.Histogram(
            bh.axis.Regular(bins=10000 - 25, start=25, stop=10000)
        )
        # can add a/e, multiplicity, qc, etc in future

        self.fig = self.canvas.fig
        self.ax = self.fig.add_subplot(111)
        self.line = self.ax.step(
            self.ehist.axes.centers[0], self.ehist.view(), where="mid"
        )[0]
        self.ax.set_xlabel("energy (keV)")
        self.ax.set_ylabel("counts")

        self.show()

    def update_hist(self):
        for _det, hit in self.ev.energy_dict.items():
            self.ehist.fill(hit)

    def plot_energy_hist(self):
        self.update_hist()
        self.line.set_ydata(self.ehist.view())
        high = np.amax(self.ehist.view())
        self.ax.set_ylim([-0.5, high * 1.2])
        self.fig.canvas.draw()


class Waveform_window(QWidget):
    exploded = False

    def __init__(self, browsers, ev):
        super().__init__()
        uic.loadUi(
            str(pathlib.Path(__file__).parent / "layouts/legend_waveform_display.ui"),
            self,
        )
        self.browsers = browsers
        self.ev = ev

        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.canvas = MplCanvas(width=10, height=10, dpi=100)
        self.update_strings()
        self.param_select.addItems(["wf_blsub", "wf_pz", "wf_trap", "curr"])

        self.plot_type.currentTextChanged.connect(self.update_plot)
        self.param_select.currentTextChanged.connect(self.update_plot)

        self.explode_toggle.clicked.connect(self.explode_plot)
        # self.duplicate_button.clicked.connect(self.plot_next_event)
        self.plot.removeWidget(self.fake)
        self.fake.deleteLater()
        self.fake = None

        self.plot.addWidget(self.canvas)
        self.show()
        self.update_plot()

    def update_strings(self):
        self.plot_type.clear()
        strings = [
            f"String:{string:02}"
            for string in np.unique(
                list(self.ev.chmap.map("location.string", unique=False))
            )
        ]
        self.plot_type.addItems(["all", "above threshold", *strings])

    def explode_plot(self):
        self.exploded = ~self.exploded
        if self.exploded == -1:
            self.explode_toggle.setText("Compress")
        else:
            self.explode_toggle.setText("Explode")
        self.update_plot()

    def update_plot(self):
        self.browsers.get_browsers()
        for i in reversed(range(self.plot.count())):
            self.plot.itemAt(i).widget().setParent(None)
        self.canvas = MplCanvas(width=10, height=10, dpi=100)
        self.plot.addWidget(self.canvas)
        if self.exploded is False:
            plot_event_compressed(
                self.ev,
                self.canvas.fig,
                self.browsers.browsers,
                line_parameter=self.param_select.currentText(),
                plot_category=self.plot_type.currentText(),
            )
        else:
            plot_event_exploded(
                self.ev,
                self.canvas.fig,
                self.browsers.browsers,
                line_parameter=self.param_select.currentText(),
                plot_category=self.plot_type.currentText(),
            )
        self.canvas.draw()


class MainWindow(QtWidgets.QMainWindow):
    w = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(
            str(pathlib.Path(__file__).parent / "layouts/legend_event_display2.ui"),
            self,
        )
        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.canvas = MplCanvas(width=10, height=10, dpi=100)
        self.ev = event_viewer(self.canvas.fig)
        self.wf_browsers = event_waveform_browser(self.ev)

        self.period_select.addItems(self.get_possible_periods())
        self.run_select.addItems(["*"])
        self.cycle_select.addItems(["*"])
        self.period_select.currentTextChanged.connect(self.update_runs)
        self.run_select.currentTextChanged.connect(self.update_cycles)

        self.plot_event.clicked.connect(self.update_plot)
        self.next_event.clicked.connect(self.plot_next_event)
        self.previous_event.clicked.connect(self.plot_previous_event)
        self.plot_waveforms.clicked.connect(self.new_waveform_window)
        self.play_run.clicked.connect(self.playback_run)
        self.plot.addWidget(self.canvas)

        self.wf_window = None
        self.ehist_window = None
        self.show()
        self.initialise_wfs()

    def get_possible_periods(self):
        hit_path = self.ev.config["tier_hit"]
        pers = ["*"]
        pers += sorted(os.listdir(pathlib.Path(hit_path) / "phy"))
        return pers

    def update_runs(self, s):
        self.run_select.setCurrentIndex(0)
        self.run_select.clear()
        hit_path = self.ev.config["tier_hit"]
        runs = ["*"]
        runs += sorted(os.listdir(pathlib.Path(hit_path) / "phy" / s))
        self.run_select.addItems(runs)

    def update_cycles(self, s):
        self.cycle_select.setCurrentIndex(0)
        self.cycle_select.clear()
        hit_path = self.ev.config["tier_hit"]
        cycles = ["*"]
        if s == "*":
            self.play_run.setEnabled(False)
        else:
            self.play_run.setEnabled(True)
            with contextlib.suppress(Exception):
                cycles += sorted(
                    [
                        file.split("-")[4]
                        for file in os.listdir(
                            pathlib.Path(hit_path)
                            / "phy"
                            / self.period_select.currentText()
                            / s
                        )
                    ]
                )
        self.cycle_select.addItems(cycles)

    def update_plot(self):
        self.ev.plot_event(
            self.period_select.currentText(),
            self.run_select.currentText(),
            self.cycle_select.currentText(),
            int(self.index_choose.text()),
        )
        self.canvas.draw()
        if self.wf_window is not None and self.wf_window.isVisible():
            self.wf_window.update_plot()
        if self.ehist_window is not None:
            self.ehist_window.plot_energy_hist()

    def plot_next_event(self):
        self.index_choose.setText(str(int(self.index_choose.text()) + 1))
        self.update_plot()

    def plot_previous_event(self):
        self.index_choose.setText(str(int(self.index_choose.text()) - 1))
        self.update_plot()

    def initialise_wfs(self):
        self.thread = QThread()
        self.browser_creator = getBrowsers(self.wf_browsers)
        self.browser_creator.moveToThread(self.thread)
        self.thread.started.connect(self.browser_creator.run)
        self.browser_creator.finished.connect(self.thread.quit)
        self.browser_creator.finished.connect(self.browser_creator.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.plot_waveforms.setEnabled(False)
        self.thread.finished.connect(lambda: self.plot_waveforms.setEnabled(True))

    def update_wf_browsers(self):
        if self.wf_browsers is None:
            self.wf_browsers = event_waveform_browser(self.ev)
            self.inititalise_wfs()
            # self.wf_browsers = event_waveform_browser(ev=self.ev)
        else:
            self.wf_browsers.get_browsers()

    def new_waveform_window(self):
        self.update_wf_browsers()
        self.wf_window = Waveform_window(self.wf_browsers, self.ev)
        self.wf_window.show()

    def closeEvent(self):
        if self.w:
            self.w.close()

    def pause(self):
        self.paused = True

    def playback_run(self):
        self.run_select.setEnabled(False)
        self.period_select.setEnabled(False)
        self.cycle_select.setEnabled(False)

        self.plot_waveforms.setEnabled(False)
        self.plot_event.setEnabled(False)
        self.next_event.setEnabled(False)
        self.previous_event.setEnabled(False)
        self.play_run.setText("Pause")
        self.paused = False
        self.play_run.disconnect()
        self.play_run.clicked.connect(self.pause)
        if self.ehist_window is None:
            self.ehist_window = ehist_window(self.ev)
        if not self.ehist_window.isVisible():
            self.ehist_window.show()

        while self.paused is False:
            self.plot_next_event()
            time.sleep(0.001)
            QApplication.processEvents()

        self.run_select.setEnabled(True)
        self.period_select.setEnabled(True)
        self.cycle_select.setEnabled(True)

        self.plot_waveforms.setEnabled(True)
        self.plot_event.setEnabled(True)
        self.next_event.setEnabled(True)
        self.previous_event.setEnabled(True)
        self.play_run.setText("Play")
        self.play_run.disconnect()
        self.play_run.clicked.connect(self.playback_run)


# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     w = MainWindow()
#     app.exec_()
