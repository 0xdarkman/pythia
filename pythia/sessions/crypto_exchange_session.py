class CryptoExchangeSession:
    def __init__(self, env, agent):
        self.environment = env
        self.agent = agent
        self._start_balance = None

    def run(self):
        s = self.environment.reset()
        self._start_balance = self.agent.balance_in(self.agent.start_coin)
        done = False
        while not done:
            a = self.agent.step(s)
            s, _, done, _ = self.environment.step(a)

    def difference(self):
        return self.agent.balance_in(self.agent.start_coin) - self._start_balance