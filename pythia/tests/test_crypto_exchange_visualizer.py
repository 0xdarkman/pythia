import pytest
import matplotlib.pyplot as plt
from decimal import Decimal

from pythia.tests.crypto_doubles import RatesStub, RecordsStub, entry
from pythia.core.visualization.coin_exchange_visualizer import CoinExchangeVisualizer


class PlotSpy:
    def __init__(self):
        self.is_shown = False
        self.line = None
        self.title_str = None
        self.x_axis = None
        self.y_axis = None
        self.plots = list()

    def show(self):
        self.is_shown = True

    def plot(self, *args, **kwargs):
        self.plots.append((args, kwargs))

    def title(self, s, *args, **kwargs):
        self.title_str = s

    def xlabel(self, s, *args, **kwargs):
        self.x_axis = s

    def ylabel(self, s, *args, **kwargs):
        self.y_axis = s


@pytest.fixture
def rates():
    s = RecordsStub()
    yield RatesStub(s)
    s.close()


def mock_plot():
    spy = PlotSpy()
    plt.show = spy.show
    plt.plot = spy.plot
    plt.title = spy.title
    plt.xlabel = spy.xlabel
    plt.ylabel = spy.ylabel

    return spy


def make_visualizer(rates):
    plotter = mock_plot()
    return CoinExchangeVisualizer(rates), plotter


def test_nothing_to_visualize(rates):
    vis, plotter = make_visualizer(rates)
    vis.render("BTC_ETH")
    assert plotter.is_shown is False


def test_visualize_rates(rates):
    rates.add_record(entry("BTC_ETH", "1")).add_record(entry("BTC_ETH", "2")).finish()
    vis, plotter = make_visualizer(rates)
    vis.render("BTC_ETH")
    assert plotter.is_shown is True
    assert plotter.plots[0] == (([Decimal('1'), Decimal('2')],), {'color': 'blue', 'label': 'Rate'})


def test_visualize_meta_data(rates):
    rates.add_record(entry("BTC_ETH", "1")).add_record(entry("BTC_ETH", "2")).finish()
    vis, plotter = make_visualizer(rates)
    vis.render("BTC_ETH")
    assert plotter.title_str == "BTC_ETH"
    assert plotter.x_axis == "Time"
    assert plotter.y_axis == "Rate"


def test_visualize_recorded_short_exchange(rates):
    rates.add_record(entry("BTC_ETH", "1")).add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).finish()
    vis, plotter = make_visualizer(rates)
    vis.record_exchange(1, "ETH")
    vis.render("BTC_ETH")
    assert plotter.plots[0] == (
    (1, Decimal('2'), 'o',), {'markerfacecolor': 'r', 'markeredgecolor': 'k', 'markersize': 10})


def test_visualize_recorded_long_exchange(rates):
    rates.add_record(entry("BTC_ETH", "1")).add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).finish()
    vis, plotter = make_visualizer(rates)
    vis.record_exchange(2, "BTC")
    vis.render("BTC_ETH")
    assert plotter.plots[0] == (
    (2, Decimal('3'), '^',), {'markerfacecolor': 'g', 'markeredgecolor': 'k', 'markersize': 10})


def test_coin_not_part_of_render_exchange(rates):
    rates.add_record(entry("BTC_ETH", "1")).add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).finish()
    vis, plotter = make_visualizer(rates)
    vis.record_exchange(1, "RPL")
    vis.render("BTC_ETH")
    assert plotter.plots[0] == (([Decimal('1'), Decimal('2'), Decimal('3')],), {'color': 'blue', 'label': 'Rate'})


def test_long_and_short(rates):
    rates.add_record(entry("BTC_ETH", "1")).add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).finish()
    vis, plotter = make_visualizer(rates)
    vis.record_exchange(0, "ETH")
    vis.record_exchange(2, "BTC")
    vis.render("BTC_ETH")
    assert plotter.plots[0] == (
    (0, Decimal('1'), 'o',), {'markerfacecolor': 'r', 'markeredgecolor': 'k', 'markersize': 10})
    assert plotter.plots[1] == (
    (2, Decimal('3'), '^',), {'markerfacecolor': 'g', 'markeredgecolor': 'k', 'markersize': 10})
    assert plotter.plots[2] == (([Decimal('1'), Decimal('2'), Decimal('3')],), {'color': 'blue', 'label': 'Rate'})

