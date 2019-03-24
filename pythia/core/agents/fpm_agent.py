import os


class FpmAgent:
    model_file_name = "model.ckpt"
    memory_file_name = "memory.npz"

    def __init__(self, ann, memory, random_gen, config, logger):
        self._ann = ann
        self._memory = memory
        self._random_generator = random_gen
        self._previous_portfolio = config["setup"]["initial_portfolio"]
        self._batch_size = config["training"]["batch_size"]
        self._logger = logger

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
        model = os.path.join(path, self.model_file_name)
        memory = os.path.join(path, self.memory_file_name)
        self._logger.info("Saving agent model to: {}".format(model))
        self._logger.info("Saving agent memory to: {}".format(memory))
        self._ann.save(model)
        self._memory.save(memory)

    def restore(self, path):
        model = os.path.join(path, self.model_file_name)
        memory = os.path.join(path, self.memory_file_name)
        self._logger.info("Restoring agent model from: {}".format(model))
        self._logger.info("Restoring agent memory from: {}".format(memory))
        self._ann.restore(model)
        self._memory.restore(memory)
