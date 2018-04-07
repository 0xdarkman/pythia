from pythia.core.environment.environment_wrappers import TradingEnvironmentTableWrapper
from pythia.core.reinforcement.reward_functions.q_table import QTable

# env = gym.make('FrozenLake-v0')
env = TradingEnvironmentTableWrapper(1000.0, "tests/test_model_data.csv", 0)

Q = QTable(range(0, env.action_space.n))
lr = 0.8
y = 0.99
num_episodes = 1000000

for i in range(num_episodes):
    s = env.reset()
    d = False
    while not d:
        a = Q.epsilon_greedy_action_of(s, (1000. / (i + 1)))
        s1, r, d, _ = env.step(a)
        Q[s, a] = Q[s, a] + lr * (r + y * Q.max_value_of(s1) - Q[s, a])
        s = s1

Q.storage.print_all_sorted_by(2)

num_test_episodes = 1
rList = []
for i in range(num_test_episodes):
    s = env.reset()
    rAll = 0
    d = False
    while not d:
        a = Q.greedy_action_of(s)
        s1, r, d, _ = env.step(a)
        rAll += r
        s = s1
    rList.append(rAll)

env.render()

print("Score over time: " + str(sum(rList) / num_test_episodes))
