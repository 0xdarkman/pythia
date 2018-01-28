class PsychicTrader:
    def __init__(self, portfolio, period, threshold):
        self.portfolio = portfolio
        self.period = period
        self.threshold = threshold
        self.shares = 0
        self.actions = []
        self.perform_trading()

    def perform_trading(self):
        for time in range(len(self.period)):
            if self.has_finished(time):
                self.sell_all_shares(time)
            elif self.is_base(time):
                self.buy_max_shares(time)
            elif self.is_peak(time):
                self.sell_all_shares(time)

            if self.no_action_took_place(time):
                self.actions.append(0)

    def has_finished(self, time):
        return time == len(self.period) - 1

    def sell_all_shares(self, time):
        if self.shares > 0:
            self.actions.append(2)

        self.portfolio += self.shares * self.period[time]
        self.shares = 0

    def is_peak(self, time):
        current_price = self.period[time]
        next_price = self.period[time + 1]
        return (next_price < current_price) and (self.is_above_threshold(current_price, next_price))

    def is_above_threshold(self, big_value, small_value):
        return (big_value - small_value) / small_value >= self.threshold

    def is_base(self, time):
        next_price = self.period[time + 1]
        current_price = self.period[time]
        return next_price > current_price and (self.is_above_threshold(next_price, current_price))

    def buy_max_shares(self, time):
        num_shares = int(self.portfolio / self.period[time])
        self.portfolio -= num_shares * self.period[time]
        self.shares += num_shares

        if num_shares > 0:
            self.actions.append(1)

    def no_action_took_place(self, time):
        return len(self.actions) <= time
