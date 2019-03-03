class FpmAgent:
    def __init__(self, ann, memory, random_gen, config):
        self._ann = ann
        self._memory = memory
        self._random_generator = random_gen
        self._previous_portfolio = config["setup"]["initial_portfolio"]
        self._batch_size = config["training"]["batch_size"]

    def step(self, prices):
        self._memory.record(prices, self._previous_portfolio)
        self._train()
        return self._act()

    def _train(self):
        b = self._memory.get_random_batch(self._batch_size)
        if b.empty:
            return

        p = self._ann.train((b.prices, b.weights), b.future)
        b.predictions = p
        self._memory.update(b)

    def _act(self):
        if self._memory.ready():
            return self._next_action_from(self._ann.predict(*self._memory.get_latest()))
        else:
            return self._next_action_from(self._random_generator())

    def _next_action_from(self, action):
        self._previous_portfolio = action
        return self._previous_portfolio

    def save(self, path):
        self._ann.save(path)

    def restore(self, path):
        self._ann.restore(path)
