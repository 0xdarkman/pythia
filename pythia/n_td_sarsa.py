from pythia.core.environment.environment_wrappers import TradingEnvironmentTableWrapper
from pythia.core.reinforcement.e_greedy_policies import EpsilonGreedyPolicy
from pythia.core.reinforcement.n_step_sarsa import NStepSarsa
from pythia.core.reinforcement.q_table import QTable

# env = gym.make('FrozenLake-v0')
env = TradingEnvironmentTableWrapper(1000.0, "tests/test_stock_data.csv", 0)

action_space = range(0, env.action_space.n)
Q = QTable(action_space)
episode = 0
tdn = NStepSarsa(env, Q, EpsilonGreedyPolicy(Q, lambda: (1000. / (episode + 1))), action_space)
tdn.steps = 1
tdn.gamma = 0.99
tdn.alpha = 0.8

num_episodes = 1000000
for _ in range(num_episodes):
    tdn.run()
    episode += 1

Q.storage.print_all_sorted_by(2)

num_test_episodes = 1
rList = []
greedy_policy = EpsilonGreedyPolicy(Q, 0)
for i in range(num_test_episodes):
    s = env.reset()
    rAll = 0
    d = False
    while not d:
        a = greedy_policy.select(s)
        s1, r, d, _ = env.step(a)
        rAll += r
        s = s1
    rList.append(rAll)

env.render()

print("Score over time: " + str(sum(rList) / num_test_episodes))
