from collections import OrderedDict
from math import sqrt

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

import skrf
from . import smith_chart, util


class NetworkPlotWidget(QtWidgets.QWidget):
    S_VALS = OrderedDict((
        ("decibels", "db"),
        ("magnitude", "mag"),
        ("phase (deg)", "deg"),
        ("phase unwrapped (deg)", "deg_unwrap"),
        ("phase (rad)", "rad"),
        ("phase unwrapped (rad)", "rad_unwrap"),
        ("real", "re"),
        ("imaginary", "im"),
    ))
    S_UNITS = list(S_VALS.keys())

    def __init__(self, parent=None, **kwargs):
        super(NetworkPlotWidget, self).__init__(parent)

        self.checkBox_useCorrected = QtWidgets.QCheckBox()
        self.checkBox_useCorrected.setText("Plot Corrected")
        self.checkBox_useCorrected.setEnabled(False)

        self.comboBox_primarySelector = QtWidgets.QComboBox(self)
        self.comboBox_primarySelector.addItems(("S", "Z", "Y", "A", "Smith Chart"))

        self.comboBox_unitsSelector = QtWidgets.QComboBox(self)
        self.comboBox_unitsSelector.addItems(self.S_UNITS)

        self.comboBox_traceSelector = QtWidgets.QComboBox(self)
        self.set_trace_items()
        self.comboBox_traceSelector.setCurrentIndex(0)

        self.plot_layout = pg.GraphicsLayoutWidget(self)
        self.plot_layout.sceneObj.sigMouseClicked.connect(self.graph_clicked)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.addWidget(self.checkBox_useCorrected)
        self.horizontalLayout.addWidget(self.comboBox_primarySelector)
        self.horizontalLayout.addWidget(self.comboBox_unitsSelector)
        self.horizontalLayout.addWidget(self.comboBox_traceSelector)

        self.data_info_label = QtWidgets.QLabel("Click a data point to see info")

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(3, 3, 3, 3)  # normally this will be embedded in another application
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.addWidget(self.plot_layout)
        self.verticalLayout.addWidget(self.data_info_label)

        self.checkBox_useCorrected.stateChanged.connect(self.set_use_corrected)
        self.comboBox_primarySelector.currentIndexChanged.connect(self.update_plot)
        self.comboBox_unitsSelector.currentIndexChanged.connect(self.update_plot)
        self.comboBox_traceSelector.currentIndexChanged.connect(self.update_plot)

        self.plot = self.plot_layout.addPlot()  # type: pg.PlotItem

        self._ntwk = None
        self._ntwk_corrected = None
        self._corrected_data_enabled = True
        self._use_corrected = False
        self.corrected_data_enabled = kwargs.get('corrected_data_enabled', True)

        self.plot.addLegend()
        self.plot.showGrid(True, True)
        self.plot.setLabel("bottom", "frequency", units="Hz")

        self.last_plot = "rectangular"

    def get_use_corrected(self):
        return self._use_corrected

    def set_use_corrected(self, val):
        if val in (1, 2):
            self._use_corrected = True
        else:
            self._use_corrected = False
        self.update_plot()

    use_corrected = property(get_use_corrected, set_use_corrected)

    @property
    def ntwk(self): return self._ntwk

    @ntwk.setter
    def ntwk(self, ntwk):
        if ntwk is None or isinstance(ntwk, skrf.Network) or type(ntwk) in (list, tuple):
            self.set_trace_items(ntwk)
            self._ntwk = ntwk
            self.update_plot()
        else:
            raise TypeError("must set to skrf.Network, list of Networks, or None")

    @property
    def ntwk_corrected(self): return self._ntwk_corrected

    @ntwk_corrected.setter
    def ntwk_corrected(self, ntwk):
        if ntwk is None or isinstance(ntwk, skrf.Network) or type(ntwk) in (list, tuple):
            self.set_trace_items(ntwk)
            self._ntwk_corrected = ntwk
            self.update_plot()
        else:
            raise TypeError("must set to skrf.Network, list of Networks, or None")

    @property
    def corrected_data_enabled(self):
        return self._corrected_data_enabled

    @corrected_data_enabled.setter
    def corrected_data_enabled(self, enabled):
        if enabled is True:
            self._corrected_data_enabled = True
            self.checkBox_useCorrected.setEnabled(True)
        else:
            self._corrected_data_enabled = False
            self._use_corrected = False
            self.checkBox_useCorrected.setEnabled(False)

    def set_networks(self, ntwk, ntwk_corrected=None):
        if ntwk is None or isinstance(ntwk, skrf.Network) or type(ntwk) in (list, tuple):
            self._ntwk = ntwk
            self.set_trace_items(self._ntwk)
            if ntwk is None:
                self._ntwk_corrected = None
                self.set_trace_items(self._ntwk)
                return
        else:
            raise TypeError("must set to skrf.Network, list of Networks, or None")

        if ntwk_corrected is None or isinstance(ntwk_corrected, skrf.Network) or type(ntwk_corrected) in (list, tuple):
            self._ntwk_corrected = ntwk_corrected
        else:
            raise TypeError("must set to skrf.Network, list of Networks, or None")

        self.update_plot()

    def reset_plot(self, smith=False):
        self.plot.clear()

        if not smith and self.last_plot == "smith":
            self.plot.setAspectLocked(False)
            self.plot.autoRange()
            self.plot.enableAutoRange()
            self.plot.setLabel("bottom", "frequency", units="Hz")

        if smith and not self.last_plot == "smith":
            self.last_plot = "smith"
            self.ZGrid = smith_chart.gen_z_grid()
            self.s_unity_circle = smith_chart.gen_s_unity_circle()
            self.plot_layout.removeItem(self.plot)
            self.plot = self.plot_layout.addPlot()
            self.plot.setAspectLocked()
            self.plot.setXRange(-1, 1)
            self.plot.setYRange(-1, 1)

        if smith:
            self.plot.addItem(self.s_unity_circle)
            self.plot.addItem(self.ZGrid)

        if not smith:
            self.plot.setLabel("left", "")

        self.plot.setTitle(None)
        legend = self.plot.legend
        if legend is not None:
            legend.scene().removeItem(legend)
        self.plot.legend = None
        self.plot.addLegend()

    def clear_plot(self):
        self._ntwk = None
        self._ntwk_corrected = None
        self._ntwk_list = None
        self.reset_plot()

    def set_trace_items(self, ntwk=None):
        self.comboBox_traceSelector.blockSignals(True)
        current_index = self.comboBox_traceSelector.currentIndex()
        nports = 0

        if isinstance(ntwk, skrf.Network):
            nports = ntwk.nports
        elif type(ntwk) in (list, tuple):
            for n in ntwk:
                if n.nports > nports:
                    nports = n.nports

        self.comboBox_traceSelector.clear()
        self.comboBox_traceSelector.addItem("all")

        for n in range(nports):
            for m in range(nports):
                self.comboBox_traceSelector.addItem("S{:d}{:d}".format(m + 1, n + 1))

        if current_index <= self.comboBox_traceSelector.count():
            self.comboBox_traceSelector.setCurrentIndex(current_index)
        else:
            self.comboBox_traceSelector.setCurrentIndex(0)
        self.comboBox_traceSelector.blockSignals(False)

    def graph_clicked(self, ev):
        """
        :type ev: pg.GraphicsScene.mouseEvents.MouseClickEvent
        :return:
        """
        xy = self.plot.vb.mapSceneToView(ev.scenePos())
        if not ev.isAccepted():
            if "smith" in self.comboBox_primarySelector.currentText().lower():
                S11 = xy.x() + 1j * xy.y()
                Z = (1 + S11) / (1 - S11)
                self.data_info_label.setText(
                    "Sre: {:g}, Sim: {:g}  -  R: {:g}, X: {:g}".format(xy.x(), xy.y(), Z.real, Z.imag))
            else:
                self.data_info_label.setText("x: {:g}, y: {:g}".format(xy.x(), xy.y()))
        elif isinstance(ev.acceptedItem, pg.PlotCurveItem):
            curve = ev.acceptedItem  # type: pg.PlotCurveItem
            spoint = xy.x() + 1j * xy.y()
            sdata = curve.xData + 1j * curve.yData
            index = np.argmin(np.abs(sdata - spoint))
            frequency = curve.ntwk.frequency.f_scaled[index]
            S11 = curve.xData[index] + 1j * curve.yData[index]
            Z = (1 + S11) / (1 - S11)
            self.data_info_label.setText(
                "Freq: {:g} ({:s}), S(re): {:g}, S(im): {:g}  -  R: {:g}, X: {:g}".format(
                    frequency, curve.ntwk.frequency.unit, S11.real, S11.imag, Z.real, Z.imag))

    def update_plot(self):
        if self.corrected_data_enabled:
            if self.ntwk_corrected:
                self.checkBox_useCorrected.setEnabled(True)
            else:
                self.checkBox_useCorrected.setEnabled(False)

        if "smith" in self.comboBox_primarySelector.currentText().lower():
            self.plot_smith()
        else:
            self.plot_ntwk()
            self.last_plot = "rectangular"

    def plot_ntwk(self):
        if self.use_corrected and self.ntwk_corrected is not None:
            ntwk = self.ntwk_corrected
        else:
            ntwk = self.ntwk

        if ntwk is None:
            return
        elif type(ntwk) in (list, tuple):
            self.plot_ntwk_list()
            return

        self.reset_plot()
        self.plot.showGrid(True, True)
        self.plot.setLabel("bottom", "frequency", units="Hz")

        colors = util.trace_color_cycle(ntwk.s.shape[1] ** 2)

        trace = self.comboBox_traceSelector.currentIndex()
        n_ = m_ = 0
        if trace > 0:
            mn = trace - 1
            nports = int(sqrt(self.comboBox_traceSelector.count() - 1))
            m_ = mn % nports
            n_ = int((mn - mn % nports) / nports)

        primary = self.comboBox_primarySelector.currentText().lower()
        s_units = self.comboBox_unitsSelector.currentText()
        attr = primary + "_" + self.S_VALS[s_units]
        s = getattr(ntwk, attr)
        for n in range(ntwk.s.shape[2]):
            for m in range(ntwk.s.shape[1]):
                if trace > 0:
                    if not n == n_ or not m == m_:
                        continue
                c = next(colors)
                label = "S{:d}{:d}".format(m + 1, n + 1)

                if "db" in attr:
                    splot = pg.PlotDataItem(pen=pg.mkPen(c), name=label)
                    if not np.any(s[:, m, n] == -np.inf):
                        splot.setData(ntwk.f, s[:, m, n])
                    self.plot.addItem(splot)
                else:
                    self.plot.plot(ntwk.f, s[:, m, n], pen=pg.mkPen(c), name=label)
        self.plot.setLabel("left", s_units)
        self.plot.setTitle(ntwk.name)

    def plot_ntwk_list(self):
        if self.use_corrected and self.ntwk_corrected is not None:
            ntwk_list = self.ntwk_corrected
        else:
            ntwk_list = self.ntwk

        if ntwk_list is None:
            return

        self.reset_plot()
        self.plot.showGrid(True, True)
        self.plot.setLabel("bottom", "frequency", units="Hz")

        colors = util.trace_color_cycle()

        trace = self.comboBox_traceSelector.currentIndex()
        n_ = m_ = 0
        if trace > 0:
            mn = trace - 1
            nports = int(sqrt(self.comboBox_traceSelector.count() - 1))
            m_ = mn % nports
            n_ = int((mn - mn % nports) / nports)

        primary = self.comboBox_primarySelector.currentText().lower()
        s_units = self.comboBox_unitsSelector.currentText()
        attr = primary + "_" + self.S_VALS[s_units]

        for ntwk in ntwk_list:
            s = getattr(ntwk, attr)
            for n in range(ntwk.s.shape[2]):
                for m in range(ntwk.s.shape[1]):
                    if trace > 0:
                        if not n == n_ or not m == m_:
                            continue
                    c = next(colors)
                    label = ntwk.name
                    if ntwk.s.shape[1] > 1:
                        label += " - S{:d}{:d}".format(m + 1, n + 1)

                    if "db" in attr:
                        splot = pg.PlotDataItem(pen=pg.mkPen(c), name=label)
                        if not np.any(s[:, m, n] == -np.inf):
                            splot.setData(ntwk.f, s[:, m, n])
                        self.plot.addItem(splot)
                    else:
                        self.plot.plot(ntwk.f, s[:, m, n], pen=pg.mkPen(c), name=label)
        self.plot.setLabel("left", s_units)

    def plot_smith(self):
        if self.use_corrected and self.ntwk_corrected is not None:
            ntwk = self.ntwk_corrected
        else:
            ntwk = self.ntwk

        if ntwk is None:
            self.reset_plot(smith=True)
            return
        elif type(ntwk) in (list, tuple):
            self.plot_smith_list()
            return

        self.reset_plot(smith=True)

        colors = util.trace_color_cycle(ntwk.s.shape[1] ** 2)

        trace = self.comboBox_traceSelector.currentIndex()
        n_ = m_ = 0
        if trace > 0:
            mn = trace - 1
            nports = int(sqrt(self.comboBox_traceSelector.count() - 1))
            m_ = mn % nports
            n_ = int((mn - mn % nports) / nports)

        for n in range(ntwk.s.shape[2]):
            for m in range(ntwk.s.shape[1]):
                if trace > 0:
                    if not n == n_ or not m == m_:
                        continue
                c = next(colors)
                label = "S{:d}{:d}".format(m + 1, n + 1)

                s = ntwk.s[:, m, n]
                curve = self.plot.plot(s.real, s.imag, pen=pg.mkPen(c), name=label)
                curve.curve.setClickable(True)
                curve.curve.ntwk = ntwk
        self.plot.setTitle(ntwk.name)

    def plot_smith_list(self):
        self.reset_plot(smith=True)

        ntwk_list = self.ntwk
        if ntwk_list is None:
            return

        colors = util.trace_color_cycle()

        trace = self.comboBox_traceSelector.currentIndex()
        n_ = m_ = 0
        if trace > 0:
            mn = trace - 1
            nports = int(sqrt(self.comboBox_traceSelector.count() - 1))
            m_ = mn % nports
            n_ = int((mn - mn % nports) / nports)

        for ntwk in ntwk_list:
            for n in range(ntwk.s.shape[2]):
                for m in range(ntwk.s.shape[1]):
                    if trace > 0:
                        if not n == n_ or not m == m_:
                            continue
                    c = next(colors)
                    label = ntwk.name
                    if ntwk.s.shape[1] > 1:
                        label += " - S{:d}{:d}".format(m + 1, n + 1)

                    s = ntwk.s[:, m, n]
                    curve = self.plot.plot(s.real, s.imag, pen=pg.mkPen(c), name=label)
                    curve.curve.setClickable(True)
                    curve.curve.ntwk = ntwk
