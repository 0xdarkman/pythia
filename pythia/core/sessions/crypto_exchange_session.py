class CryptoExchangeSession:
    def __init__(self, env, agent):
        """
        Simple run through a crypto coin exchange session with the provided environment and agent.

        :param env: A crypto exchange environment providing an OpenAI gym environment like interface
        :param agent: An agent acting within the crypto exchange environment
        """
        self.environment = env
        self.agent = agent
        self._start_coin = env.coin
        self._start_balance = None

    def run(self):
        s = self.environment.reset()
        self._start_balance = self.environment.balance_in(self._start_coin)
        done = False
        while not done:
            a = self.agent.step(s)
            s, _, done, _ = self.environment.step(a)

    def difference(self):
        return self.environment.balance_in(self._start_coin) - self._start_balance
