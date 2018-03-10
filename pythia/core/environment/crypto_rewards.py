class TotalBalanceReward:
    def __call__(self, env):
        return env.normalized_balance


class RatesChangeReward:
    def __call__(self, env):
        idx = env.coin_to_index[env.coin]
        return env.state[2 + idx] - env.prev_state[2 + idx]