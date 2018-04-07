import os
import time

from pythia.core.environment.environment_wrappers import TradingEnvironmentTableWrapper
from reinforcement.policies.e_greedy_policies import EpsilonGreedyPolicy, NormalEpsilonGreedyPolicy
from pythia.core.n_step_sarsa import NStepSarsa
from reinforcement.reward_functions.q_neuronal import QNeuronal
from reinforcement.models.q_regression_model import QRegressionModel

# env = gym.make('FrozenLake-v0')
env = TradingEnvironmentTableWrapper(1000.0, "data/sxe.csv", 0)
model_dir = "reinforcement/model_data"
for f in os.listdir(model_dir):
    os.remove(os.path.join(model_dir, f))

model = QRegressionModel(env.action_space.n + 1, [100], 0.01)
Q = QNeuronal(model,  env.action_space.n, 10)
episode = 0
tdn = NStepSarsa(env, Q, NormalEpsilonGreedyPolicy(lambda: (100 / (episode + 1))), Q.action_space)
tdn.steps = 1
tdn.gamma = 0.99
tdn.alpha = 0.8

start = time.perf_counter()
num_episodes = 10000
for _ in range(num_episodes):
    tdn.run()
    episode += 1
end = time.perf_counter()
print(end - start)

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
