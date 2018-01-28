import pickle
import random

import numpy as np

from pythia.environment.environment_wrappers import TradingEnvironment

env = TradingEnvironment(2000.0, "tests/test_stock_data.csv")

number_actions = env.action_space.n
number_states = env.observation_space.n - 1
Q = np.zeros([number_states, number_actions])
C = np.zeros([number_states, number_actions])

num_episodes = 10000
gamma = 0.5

for i in range(num_episodes):
    episode_data = []
    env.reset()
    done = False
    while not done:
        a = random.randint(0, 2)
        _, r, done, _ = env.step((a - 1) * 0.3)
        episode_data.append((a, r))

    total_reward = 0
    for s in reversed(range(0, number_states)):
        total_reward = gamma * total_reward + episode_data[s + 1][1]
        a = episode_data[s][0]
        C[s, a] += 1
        Q[s, a] = Q[s, a] + (1.0 / C[s, a]) * (total_reward - Q[s, a])

print(Q)

rewards = []
done = False
env.reset()
for s in range(0, number_states):
    a = np.argmax(Q[s, :])
    _, r, done, _ = env.step((a - 1) * 0.3)

    rewards.append(r)
    if done:
        break

print("Score over time: " + str(sum(rewards)))

env.render()

qFile = open(r"models/qtable.pkl", "wb")
pickle.dump(Q, qFile)
qFile.close()

