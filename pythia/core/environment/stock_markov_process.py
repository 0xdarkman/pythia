class StockMarkovProcess(object):
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.all_states = self.make_all_states()

    def make_all_states(self):
        states = []
        for i in range(0, self.stock_data.number_of_stocks()):
            step_states = []
            for state in self.get_states_of_step(i):
                step_states.append(state)

            states.append(step_states)

        return states

    def get_states_of_step(self, step):
        price = self.get_price_of_step(step)
        states = [[price, 0, step, 0]]
        for i in range(0, step):
            prev_states = self.get_states_of_step(i)
            for prev in prev_states:
                if prev[1] > 0:
                    self.add_only_if_not_present(states, [price, 0, step, prev[3] + (prev[0] - prev[1])])
                elif prev[3] > 0:
                    self.add_only_if_not_present(states, [price, 0, step, prev[3]])
                    self.add_only_if_not_present(states, [price, prev[0], step, prev[3]])

        for i in range(0, step):
            prev_state = [price, self.get_price_of_step(i), step, 0]
            if prev_state in states:
                continue

            states.append(prev_state)

        return states

    def add_only_if_not_present(self, states, state):
        if state not in states:
            states.append(state)

    def get_price_of_step(self, step):
        return self.stock_data.data[step][0]

    def get_all_states(self):
        return [states for sublist in self.all_states for states in sublist]

    def action_at_state(self, state, action):
        step = state[2]
        bought_shares = state[1]
        reward = 0

        if step + 1 == len(self.all_states):
            if action == -1 and bought_shares != 0:
                reward = state[0] - bought_shares
            return reward, [], True

        state_idx = 0
        next_states = self.all_states[step + 1]

        if action == -1 and bought_shares != 0:
            reward = next_states[0][0] - bought_shares
        elif action == 0 or bought_shares != 0:
            state_idx = self.all_states[step].index(state)
        elif action == 1:
            for i, next_state in enumerate(next_states):
                if next_state[1] == state[0]:
                    state_idx = i
                    break

        return reward, next_states[state_idx], False
