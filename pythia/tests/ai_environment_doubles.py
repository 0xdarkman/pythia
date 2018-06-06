class RewardCalculatorStub:
    def __init__(self, reward):
        self.reward = reward

    def __call__(self, env):
        return self.reward


class RewardCalculatorSpy:
    def __init__(self):
        self.received_arg = None

    def __call__(self, env):
        self.received_arg = env