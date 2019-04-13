class FpmHistoricalSeries:
    def __init__(self, *symbols):
        if len(symbols) == 0:
            raise self.NoSymbolsError()

        self.symbols = symbols
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        def make_price(symbol):
            s = next(symbol)[1]
            return [s.close, s.high, s.low]

        return [make_price(s) for s in self.symbol_iters]

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.symbol_iters = [s.iterrows() for s in self.symbols]

    class NoSymbolsError(AttributeError):
        pass
