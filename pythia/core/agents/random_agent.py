class RuleAgent:
    def __init__(self, generator):
        self._generator = generator

    def step(self, _):
        return self._generator()

    def save(self, _):
        pass

    def restore(self, _):
        pass
