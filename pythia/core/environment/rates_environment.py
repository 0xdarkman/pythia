from decimal import Decimal


class RatesEnvironment:
    def __init__(self, rates, start_token, start_amount):
        """
        Environment representing exchange rates. Provides exchange states containing rates, fees and other market data.
        Implements a mechanism to exchange tokens against other tokens. Keeps track of currently active token
        and a total balance of tokens

        :param rates: source stream containing market information for token exchanges
        :param start_token: token the starting balance is held in
        :param start_amount: the starting balance
        """
        self.rates_stream = rates
        self.time = 0
        self._start_amount = Decimal(start_amount)
        self.start_token = start_token
        self._amount = self._start_amount
        self._token = self.start_token
        self._current_state = None
        self._next_state = None
        self._listeners = list()
        self.reset()

    @property
    def amount(self):
        return self._amount

    @property
    def token(self):
        return self._token

    def reset(self):
        self.time = 0
        self._token = self.start_token
        self._amount = self._start_amount
        self.rates_stream.reset()
        try:
            self._current_state = next(self.rates_stream)
            self._next_state = next(self.rates_stream)
            return { "token": self.token, "balance": self.amount, "rates": self._current_state }
        except StopIteration:
            raise EnvironmentFinished("A Crypto environment needs at least 2 entries to be initialised.")

    def step(self, action):
        if self._next_state is None:
            raise EnvironmentFinished("CryptoEnvironment finished. No further steps possible.")

        if action is not None and action != self.token:
            self._exchange_token(action)

        self._move_to_next_state()
        self.time += 1
        return { "token": self.token, "balance": self.balance_in(self.start_token), "rates": self._current_state}, None, self._next_state is None, None

    def _exchange_token(self, action):
        exchange = self._get_exchange_to(action)
        self._token = action
        self._amount = (self._amount * exchange.rate) - exchange.fee
        for listener in self._listeners:
            listener(self.time, action)

    def _get_exchange_to(self, other_token):
        return self._current_state[self.token + "_" + other_token]

    def _move_to_next_state(self):
        self._current_state = self._next_state
        self._next_state = next(self.rates_stream, None)

    def balance_in(self, token):
        if self.token == token:
            return self.amount
        return self.amount if self.token == token else self.amount * self._get_exchange_to(token).rate

    def register_listener(self, listener):
        self._listeners.append(listener)


class EnvironmentFinished(StopIteration):
    pass
