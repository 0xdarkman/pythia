class RatesExchangeSession:
    def __init__(self, env, agent):
        """
        Simple run through a rates exchange session with the provided environment and agent.

        :param env: A rates exchange environment providing an OpenAI gym environment like interface
        :param agent: An agent acting within the rates exchange environment
        """
        self.environment = env
        self.agent = agent
        self._start_token = env.token
        self._start_balance = None

    def run(self):
        s = self.environment.reset()
        self._start_balance = self.environment.balance_in(self._start_token)
        a = self.agent.start(s)
        while True:
            s, r, done, _ = self.environment.step(a)
            if not done:
                a = self.agent.step(s, r)
            else:
                self.agent.finish(r)
                break

    def difference(self):
        return self.environment.balance_in(self._start_token) - self._start_balance
