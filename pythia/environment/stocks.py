import matplotlib.pyplot as plt
import pandas as pd

from pythia.environment.simulators import PsychicTrader


class ExceededStockData(Exception):
    pass


class StockData:
    def __init__(self, file):
        data_file = pd.read_csv(file)
        self.data = data_file.iloc[:, 1:].values[::-1]

    def get_period(self, from_idx, length):
        if (from_idx + length) > len(self.data):
            raise ExceededStockData

        return self.data[from_idx:from_idx + length]

    def number_of_stocks(self):
        return len(self.data)


class StockVisualizer:
    def __init__(self, period, actions):
        self.period = period
        self.actions = actions

    def plot(self):
        opening_prices = self.period[:, 0]
        markers = ['o', '^']
        colors = ['r', 'g']

        plt.plot(opening_prices, color='blue', label='Stock Price')
        for i, p in enumerate(opening_prices):
            if i == len(self.actions) + 1:
                action = 0
            else:
                action = self.actions[i + 1]

            if action != 0:
                plt.plot(i, p, markers[action - 1],
                         markerfacecolor=colors[action - 1],
                         markeredgecolor='k',
                         markersize=10)

        plt.title('Stock Price + Actions')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    data = StockData("tests/test_stock_data.csv")
    period = data.get_period(0, 20)
    trader = PsychicTrader(1000.0, period[:, 0], 0.01)
    view = StockVisualizer(period, trader.actions)
    view.plot()
