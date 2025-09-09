#
#  Copyright (C) 2024-2025
#  MIT
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
"""This module collects tests specific to bokeh."""

import os
import pytest

import numpy as np

from sherpa import plot
from sherpa.astro import ui
from sherpa.utils.testing import requires_data, requires_fits
from sherpa.plot import DataPlot
from sherpa.data import Data1D, Data1DInt


@requires_fits
@requires_data
def test_bokeh_delchi(caplog, clean_ui):
    """Test a specific multi-panel plot with the bokeh backend.

    This also implicitly tests setting a line_color, because
    delchi put a black horizontal line on the plot.

    This is a regression test for https://github.com/sherpa/sherpa/issues/1975
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "astro",
        "datastack",
        "tests",
        "data",
        "3c273.pi",
    )
    ui.load_pha(path)
    ui.set_source(1, ui.powlaw1d(name="powlaw"))

    bokehbackend = pytest.importorskip("sherpa.plot.bokeh_backend")
    newback = bokehbackend.BokehBackend()
    with plot.TemporaryPlottingBackend(newback):
        ui.plot_fit_delchi()


def test_bokeh_specific_option_does_not_error():
    """Test that a bokeh-specific option does not error out.

    """
    bokehbackend = pytest.importorskip("sherpa.plot.bokeh_backend")
    # previous line guarantees that bokeh is installed
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    data = Data1DInt('tst', np.asarray([1, 2, 3]), np.asarray([1, 2, 3]) + 1,
                      np.asarray([10, 12, 10.5]))
    pl = DataPlot()
    pl.prepare(data, stat=None)
    newback = bokehbackend.BokehBackend()
    with plot.TemporaryPlottingBackend(newback):
        pl.plot(marker='o',tags=['foo', 10], line_dash_offset=3, linestyle="dashed")
        html = file_html(plot.backend.current_fig, CDN, "my plot")

    assert '"tags":["foo",10]' in html
    assert '"line_dash_offset":3' in html


def test_bokeh_specific_option_scatter():
    bokehbackend = pytest.importorskip("sherpa.plot.bokeh_backend")
    # previous line guarantees that bokeh is installed
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    rng = np.random.default_rng(1273)
    z1 = rng.wald(100, 20, size=1000)
    z2 = rng.wald(100, 2000, size=1000)

    splot = plot.ScatterPlot()
    splot.prepare(z1, z2, xlabel='$$z_1$$', ylabel='$$z_2$$')
    newback = bokehbackend.BokehBackend()
    with plot.TemporaryPlottingBackend(newback):
        splot.plot(xlog=True, hatch_pattern='/', markersize=20, alpha=0.5)
        html = file_html(plot.backend.current_fig, CDN, "my plot")

    assert '"hatch_pattern":{"type":"value","value":"/"}' in html


def test_warning_plot_linecolor():
    """We get a warning when using linecolor: DataPlot"""
    bokehbackend = pytest.importorskip("sherpa.plot.bokeh_backend")

    data = Data1D('tst', np.asarray([1, 2, 3]), np.asarray([10, 12, 10.5]))
    pl = DataPlot()
    pl.prepare(data, stat=None)
    newback = bokehbackend.BokehBackend()
    with plot.TemporaryPlottingBackend(newback):
        with pytest.raises(AttributeError,
                           match="unexpected attribute 'keyword_make_no_sense' to Scatter"):
            pl.plot(keyword_make_no_sense='mousey')