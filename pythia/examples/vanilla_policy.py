import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from pythia.environment.environment_wrappers import TradingEnvironment

env = TradingEnvironment(2000.0, "model_data/sxe.csv")
gamma = 0.99


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


class Agent:
    def __init__(self, lr, s_size, a_size, h_size):
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + "_holder")
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


tf.reset_default_graph()

agent = Agent(lr=1e-2, s_size=5, a_size=3, h_size=8)

total_episodes = 5000
max_ep = 9999999
update_frequency = 5

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_rewards = []
    total_length = []

    grad_buffer = sess.run(tf.trainable_variables())
    for idx, grad in enumerate(grad_buffer):
        grad_buffer[idx] = 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            a_dist = sess.run(agent.output, feed_dict={agent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            a_env = 0.0
            if a == 1:
                a_env = 0.2
            elif a == 2:
                a_env = -0.2

            s1, r, d, _ = env.step(a_env)
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            if d:
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {agent.reward_holder: ep_history[:, 2], agent.action_holder: ep_history[:, 1],
                             agent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(agent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    grad_buffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(agent.gradient_holders, grad_buffer))
                    _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                    for idx, grad in enumerate(grad_buffer):
                        grad_buffer[idx] = 0

                total_rewards.append(running_reward)
                total_length.append(j)
                break

        if i % 100 == 0:
            mean_total = np.mean(total_rewards[-100:])
            print(mean_total)
        i += 1

env.render()

