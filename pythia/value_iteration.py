import numpy as np

from pythia.environment.environment_wrappers import TradingEnvironment
from pythia.environment.stocks import StockData
from pythia.reinforcement.stock_markov_process import StockMarkovProcess
from pythia.reinforcement.value_table import ValueTable

stock_data = StockData("tests/test_model_data.csv")
number_of_stocks = stock_data.number_of_stocks()
V = ValueTable()
gamma = 1.0
delta = np.infty

process = StockMarkovProcess(stock_data)

while delta > 0.00001:
    delta = 0
    for state in process.get_all_states():
        temp = V[state]
        max_value = -float("inf")
        for a in range(-1, 2):
            reward, next_state, done = process.action_at_state(state, a)
            if done:
                max_value = reward - abs(state[3])
                break
            val = reward + gamma * V[next_state]
            if val > max_value:
                max_value = val

        V[state] = max_value
        delta = max(delta, temp - V[state])

V.print_states(process.all_states)

env = TradingEnvironment(1000.0, "tests/test_model_data.csv")
rewards = []
done = False
env_state = env.reset()
step = 0

while not done:
    s = [env_state[0], env_state[-1], step, 0]
    max_value = -float("inf")
    action = 0
    for a in [0, 1, -1]:
        reward, next_state, done = process.action_at_state(s, a)
        if done:
            break

        val = reward + gamma * V[next_state]
        if val > max_value:
            max_value = val
            action = a

    env_state, r, done, _ = env.step(action)
    rewards.append(r)
    step += 1

print("Score over time: " + str(sum(rewards)))

env.render()
