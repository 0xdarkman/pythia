from collections import deque


class NStepSarsa(object):
    def __init__(self, environment, q_function, policy, action_space):
        self.env = environment
        self.q_function = q_function
        self.policy = policy
        self.action_space = action_space
        self.steps = 1
        self.gamma = 0.99
        self.alpha = 0.4
        self.epsilon = 0.1
        self.record = deque()

    def run(self):
        self.record.clear()
        s = self.env.reset()
        done = False
        a = self.policy.select(s)
        while not done:
            s1, r, done, _ = self.env.step(a)
            self.record.append((r, s, a))

            a1 = a
            while self.is_updating_from_record(done):
                G = 0
                i = 0
                for r_t, _, _ in self.record:
                    G += r_t * pow(self.gamma, i)
                    i += 1
                if not done:
                    a1 = self.policy.select(s1)
                    G += pow(self.gamma, self.steps) * self.q_function[s1, a1]

                _, s_t, a_t = self.record.popleft()
                self.q_function.learn(s_t, a_t, self.alpha * (G - self.q_function[s_t, a_t]))

            s = s1
            a = a1

    def is_updating_from_record(self, done):
        return (len(self.record) == self.steps) or (done and len(self.record) != 0)
