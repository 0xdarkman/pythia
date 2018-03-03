from decimal import Decimal


class CryptoEnvironment:
    def __init__(self, rates, start_coin, start_amount):
        """
        Environment representing crypto coin exchanges. Provides exchange states containing rates, miner fees and other
        market data. Implements a mechanism to exchange coins against other coins. Keeps track of currently active coin
        and a total balance of coins

        :param rates: source stream containing market information for coin exchanges
        :param start_coin: crypto coin the starting balance is held in
        :param start_amount: the starting balance
        """
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
            return self.coin, self._current_state
        except StopIteration:
            raise EnvironmentFinished("A Crypto environment needs at least 2 entries to be initialised.")

    def step(self, action):
        if self._next_state is None:
            raise EnvironmentFinished("CryptoEnvironment finished. No further steps possible.")

        if action is not None:
            self._exchange_coin(action)

        self._move_to_next_state()
        return (self.coin, self._current_state), None, self._next_state is None, None

    def _exchange_coin(self, action):
        exchange = self._get_exchange_to(action)
        self._coin = action
        self._amount = (self._amount * exchange.rate) - exchange.minerFee

    def _get_exchange_to(self, other_coin):
        return self._current_state[self.coin + "_" + other_coin]

    def _move_to_next_state(self):
        self._current_state = self._next_state
        self._next_state = next(self.rates_stream, None)

    def balance_in(self, coin):
        if self.coin == coin:
            return self.amount
        return self.amount if self.coin == coin else self.amount * self._get_exchange_to(coin).rate


class EnvironmentFinished(StopIteration):
    pass