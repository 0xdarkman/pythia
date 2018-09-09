class FpmSession:
    def __init__(self, env, agent, logger):
        self.env = env
        self.agent = agent
        self.log = logger
        self.log_interval = 1

    def run(self):
        num = 0
        r = None
        s = self.env.reset()
        done = False
        while True:
            a = self.agent.step(s)
            if done:
                break
            s, r, done, _ = self.env.step(a)
            if self._should_log(num):
                self.log(r)
            num += 1

        return r

    def _should_log(self, num):
        return self.log_interval > 0 and num % self.log_interval == 0