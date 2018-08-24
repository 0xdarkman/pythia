class FpmAgent:
    def __init__(self, ann, memory, config):
        self._ann = ann
        self._memory = memory
        self._previous_portfolio = config["setup"]["initial_portfolio"]
        self._batch_size = config["training"]["batch_size"]

    def step(self, prices):
        self._memory.record(prices, self._previous_portfolio)
        b = self._memory.get_random_batch(self._batch_size)
        p = self._ann.train((b.prices, b.weights), b.future)
        b.predictions = p
        self._memory.update(b)
        self._previous_portfolio = self._ann.predict(prices, self._previous_portfolio)
        return self._previous_portfolio
