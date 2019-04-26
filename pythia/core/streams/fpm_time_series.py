import os

import pandas as pd


class FpmTimeSeries:
    def __iter__(self):
        return self

    class NoSymbolsError(AttributeError):
        pass


class FpmHistoricalSeries(FpmTimeSeries):
    def __init__(self, *symbols):
        if len(symbols) == 0:
            raise self.NoSymbolsError()

        self.symbols = symbols
        self._reset_iters()

    # noinspection PyAttributeOutsideInit
    def _reset_iters(self):
        self.symbol_iters = [s.iterrows() for s in self.symbols]

    def __next__(self):
        def make_price(symbol):
            s = next(symbol)[1]
            return [s.close, s.high, s.low]

        return [make_price(s) for s in self.symbol_iters]

    def reset(self):
        self._reset_iters()
        return self.__next__()


class FpmLiveSeries(FpmTimeSeries):
    def __init__(self, connection, config):
        self.connection = connection
        self.cash = config["cash"]
        self.symbols = config["coins"]
        self.start = config["start"]
        self.data_dir = config["training_data_dir"]
        if len(self.symbols) == 0:
            raise self.NoSymbolsError()

    def __next__(self):
        try:
            def price_list_of(symbol):
                r = self.connection.get_next_prices(self.cash, symbol)
                return [r["close"], r["high"], r["low"]]

            return [price_list_of(s) for s in self.symbols]
        except TimeoutError:
            raise self.TimeoutError("The connection to query the next prices timed out")

    def reset(self):
        self.connection.reset()
        initial_prices = []
        for sym in self.symbols:
            df = pd.read_csv(os.path.join(self.data_dir, "{}_{}.csv".format(self.cash, sym)), index_col='timestamp')
            row = df.loc[self.start]
            initial_prices.append([row.close, row.high, row.low])

        return initial_prices

    class TimeoutError(TimeoutError):
        pass
