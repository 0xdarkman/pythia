import pickle

import numpy as np

from pythia.environment.environment_wrappers import TradingEnvironment

env = TradingEnvironment(2000.0, "tests/test_stock_data.csv")

number_actions = env.action_space.n
number_states = env.observation_space.n
Q = np.zeros([number_states, number_actions])
learning_rate = 0.5
gamma = 0.95
num_episodes = 5000000

rewards = []
for i in range(num_episodes):
    env.reset()
    total_reward = 0
    for j in range(number_states - 1):
        a = np.argmax(Q[j, :] + np.random.randn(1, number_actions) * (1.0 / (i+1)))
        _, r, done, _ = env.step((a - 1) * 0.3)
        Q[j, a] = Q[j, a] + learning_rate * (r + gamma * np.max(Q[j + 1, :]) - Q[j, a])
        total_reward += r
        if done:
            break
    rewards.append(total_reward)

print("Score over time: " + str(sum(rewards) / num_episodes))
print(Q)

env.render()

qFile = open(r"models/qtable.pkl", "wb")
pickle.dump(Q, qFile)
qFile.close()

