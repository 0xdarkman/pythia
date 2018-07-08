from collections import deque

from matplotlib import pyplot as plt


class CoinExchangeVisualizer:
    def __init__(self, stream):
        self.stream = stream
        self.actions = deque()

    def render(self, exchange):
        self.stream.reset()
        plt.figure(figsize=(30, 12))
        vertices = list()
        action = self._get_action()
        for i, market in enumerate(self.stream):
            rate = market[exchange].rate
            vertices.append(rate)
            if action is not None:
                action = self._mark_action(action, i, rate, exchange)

        self.stream.reset()

        if len(vertices) == 0:
            return

        self._plot_meta(exchange)
        plt.plot(vertices, color='blue', label='Rate')
        plt.show()

    def _mark_action(self, action, time, rate, exchange):
        a_t, to_coin = action
        lhs, rhs = exchange.split('_')
        if to_coin != lhs and to_coin != rhs:
            return action

        shape = 'o' if to_coin == rhs else '^'
        color = 'r' if to_coin == rhs else 'g'

        if time == a_t:
            plt.plot(a_t, rate, shape, markerfacecolor=color, markeredgecolor='k', markersize=10)
            return self._get_action()

        return action

    def _get_action(self):
        return self.actions.popleft() if len(self.actions) != 0 else None

    @staticmethod
    def _plot_meta(exchange):
        plt.title(exchange)
        plt.xlabel("Time")
        plt.ylabel("Rate")

    def record_exchange(self, time, to_coin):
        self.actions.append((time, to_coin))