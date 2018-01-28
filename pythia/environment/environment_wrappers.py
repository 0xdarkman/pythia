import math

import numpy as np

from pythia.environment.stocks import StockData, StockVisualizer

MINIMUM_ACTION = 0.05


class Discrete:
    def __init__(self, n):
        self.n = n


class TradingEnvironment:
    def __init__(self, portfolio, stock_file):
        self.portfolio = portfolio
        self.initial_portfolio = self.portfolio
        self.stock_data = StockData(stock_file)
        self.time_step = 0
        self.state = self._make_state(self.stock_data.get_period(self.time_step, 1)[0], 0, 0)
        self.shares = 0
        self.done = False
        self.actions = []
        self.action_space = Discrete(3)
        self.observation_space = Discrete(self.stock_data.number_of_stocks() - 2)
        self.previous_wealth = self.portfolio
        self.wealth = self.portfolio
        self.buying_price = 0

    def step(self, action):
        price = self.state[0]
        if self.is_buy_action(action):
            self.buy_shares(action, price)
        elif self.is_sell_action(action):
            self.sell_shares(action, price)
        else:
            self.actions.append(0)

        self.wealth = self.portfolio + self.shares * price
        reward = self._calc_reward(action)
        self.previous_wealth = self.wealth

        self.time_step += 1
        self.update_state()

        return self.state, reward, self.done, self.portfolio

    def is_buy_action(self, action):
        return action > MINIMUM_ACTION

    def is_sell_action(self, action):
        return action < -MINIMUM_ACTION

    def buy_shares(self, action, price):
        requested_value = self.portfolio * action
        bs = math.floor(requested_value / price)
        if bs > 0:
            self.buying_price = price
        self.shares += bs
        self.portfolio -= self.shares * price
        self.actions.append(1)

    def sell_shares(self, action, price):
        self.buying_price = 0.0
        requested_shares = math.floor(self.shares * math.fabs(action))
        self.portfolio += requested_shares * price
        self.shares -= requested_shares
        self.actions.append(2)

    def _calc_reward(self, action):
        return self.wealth - self.previous_wealth

    def update_state(self):
        if self.time_step + 1 == len(self.stock_data.data):
            self.done = True

        self.state = self._make_state(self.stock_data.get_period(self.time_step, 1)[0], self.shares, self.buying_price)

    def _make_state(self, prices, shares, buying_price):
        return np.append(prices, [[shares, buying_price]])

    def reset(self):
        self.portfolio = self.initial_portfolio
        self.time_step = 0
        self.buying_price = 0
        self.state = self._make_state(self.stock_data.get_period(self.time_step, 1)[0], 0, 0)
        self.shares = 0
        self.done = False
        self.previous_wealth = self.portfolio
        self.wealth = self.portfolio
        self.actions.clear()

        return self.state

    def render(self):
        vis_actions = [0] + self.actions + [0]
        visualizer = StockVisualizer(self.stock_data.get_period(0, self.time_step + 1), vis_actions)
        visualizer.plot()


class TradingEnvironmentTableWrapper(TradingEnvironment):
    def __init__(self, portfolio, stock_file, penalty):
        self.current_step = 0
        self.penalty = penalty
        super().__init__(portfolio, stock_file)

    def step(self, action):
        self.current_step += 1
        is_invalid_action = self.is_invalid_action(action)
        if action == 2:
            action = -1

        s, r, d, i = super().step(action)
        if is_invalid_action:
            r = -self.penalty
        return s, r, d, i

    def is_invalid_action(self, action):
        return (action == 1 and self.state[1] != 0) or (action == 2 and self.state[1] == 0)

    def _make_state(self, prices, shares, buying_price):
        return [float(prices[0]), float(buying_price), float(self.current_step)]

    def _calc_reward(self, action):
        if self.is_sell_action(action) and self.state[1] > 0:
            return self.state[0] - self.state[1]

        return 0.0

    def reset(self):
        self.current_step = 0
        return super().reset()


if __name__ == '__main__':
    env = TradingEnvironment(1000.0, "tests/test_stock_data.csv")
    env.step(0.0)
    env.step(1.0)
    env.step(0.0)
    env.step(0.0)
    env.step(0.0)
    env.step(-1.0)
    env.step(0.0)
    env.render()
