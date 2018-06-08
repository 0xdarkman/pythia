import sys
from io import StringIO

from pythia.core.environment.rates_ai_environment import RatesAiEnvironment
from pythia.core.environment.rates_rewards import TotalBalanceReward
from pythia.core.environment.rigged_policy import STOP_AT_THRESHOLD, RiggedPolicy
from pythia.core.streams.shape_shift_rates import ShapeShiftRates, SUPPORTED_COINS
from pythia.core.streams.rates_calculators import rates_filter
from pythia.test_integration.test_rigged_policy_makes_good_decisions import PolicyDummy, QDummy


def all_combinations():
    all = set()
    for coin_a in SUPPORTED_COINS:
        for coin_b in SUPPORTED_COINS:
            if coin_a != coin_b:
                l = coin_a + "_" + coin_b
                r = coin_b + "_" + coin_a
                if (r, l) not in all:
                    all.add((l, r))
    return all

if __name__ == "__main__":
    in_path = "../data/recordings/2018-02-28-shapeshift-exchange-records.json" if len(sys.argv) == 1 else sys.argv[1]
    #out_path = "../data/recordings/analysis/2018-02-28-shapeshift.csv" if len(sys.argv) == 1 else sys.argv[2]
    #os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #with open(in_path, 'r') as in_stream, open(out_path, 'w') as out_stream:
    #    out_stream.write(analyze(ShapeShiftRates(in_stream)).to_csv())

    max_profit = float("-inf")
    max_profit_exchange = None
    for l_ex, r_ex in all_combinations():
        with open(in_path, 'r') as in_stream:
            stream = StringIO((rates_filter(ShapeShiftRates(in_stream), [l_ex, r_ex])))
            rates = ShapeShiftRates(stream, preload=True)
            coin_a = l_ex.split('_')[0]
            coin_b = l_ex.split('_')[1]
            env = RatesAiEnvironment(rates, coin_a, "10", 1, {1: coin_a, 2: coin_b}, TotalBalanceReward())
            policy = RiggedPolicy(env, PolicyDummy(), 1.0, rigging_distance=STOP_AT_THRESHOLD, threshold=0.1)
            s = env.reset()
            start_balance = env.balance_in(coin_a)
            done = False
            while not done:
                a = policy.select(s, QDummy())
                s, _, done, _ = env.step(a)

            profit = env.balance_in(coin_a) - start_balance
            print("{} generated profit: {}".format(l_ex, profit))
            if profit > max_profit:
                max_profit = profit
                max_profit_exchange = l_ex
                print("{} has new max profit: {}".format(max_profit_exchange, max_profit))

    print("{} has the max profit: {}".format(max_profit_exchange, max_profit))
