class TotalBalanceReward:
    def __call__(self, env):
        """
        The reward is the current balance of the wallet normalized to the change from the initial balance. I.e. the
        balance has increased from 10 to 12 the reward would be 0.2.

        :param env: CryptoAiEnvironment
        :return: float representing the reward
        """
        b = env.balance_in(env.start_token)
        return float((b - env.starting_balance) / env.starting_balance)


class RatesChangeReward:
    def __call__(self, env):
        """
        The reward is difference between the current and the last rate of the active token

        :param env: CryptoAiEnvironment
        :return: float representing the reward
        """
        idx = env.token_to_index[env.token]
        return env.state[2 + idx] - env.prev_state[2 + idx]