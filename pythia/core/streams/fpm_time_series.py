class FpmTimeSeries:
    def __init__(self, cash, *symbols):
        if len(symbols) == 0:
            raise self.NoSymbolsError()

        self.cash = cash
        self.symbols = symbols
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        c = next(self.cash_iter)[1]

        def make_price(symbol):
            s = next(symbol)[1]
            return [s.close / c.close, s.high / c.high, s.low / c.low]

        return [make_price(s) for s in self.symbol_iters]

    class NoSymbolsError(AttributeError):
        pass

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.cash_iter = self.cash.iterrows()
        self.symbol_iters = [s.iterrows() for s in self.symbols]