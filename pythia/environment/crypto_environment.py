from decimal import Decimal


class CryptoEnvironment:
    def __init__(self, rates, start_coin, start_amount):
        self.rates_stream = rates
        self._amount = Decimal(start_amount)
        self._coin = start_coin
        self._current_state = None
        self._next_state = None
        self.reset()

    @property
    def amount(self):
        return self._amount

    @property
    def coin(self):
        return self._coin

    def reset(self):
        self.rates_stream.reset()
        try:
            self._current_state = next(self.rates_stream)
            self._next_state = next(self.rates_stream)
            return self._current_state
        except StopIteration:
            raise EnvironmentFinished("A Crypto environment needs at least 2 entries to be initialised.")

    def step(self, action):
        if self._next_state is None:
            raise EnvironmentFinished("CryptoEnvironment finished. No further steps possible.")

        if action is not None:
            self._exchange_coin(action)

        self._move_to_next_state()
        return self._current_state, None, self._next_state is None, None

    def _exchange_coin(self, action):
        exchange = self._current_state[self.coin + "_" + action]
        self._coin = action
        self._amount = (self._amount * exchange.rate) - exchange.minerFee

    def _move_to_next_state(self):
        self._current_state = self._next_state
        self._next_state = next(self.rates_stream, None)


class EnvironmentFinished(StopIteration):
    pass