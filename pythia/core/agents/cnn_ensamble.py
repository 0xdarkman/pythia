import tensorflow as tf
import tflearn
import numpy as np

DEFAULT_PADDING = "valid"
DEFAULT_ACTIVATION = "relu"
DEFAULT_STRIDES = [1, 1]
DEFAULT_WEIGHT_DECAY = 0.0


class CNNEnsemble:
    def __init__(self, session, assets, window_size, config):
        self.layers = []
        self.session = session
        self.config = config

        self.global_step = tf.Variable(0, trainable=False)
        self.input_prices = tf.placeholder(tf.float32, shape=[None, 3, assets, window_size])
        self.input_prev_omega = tf.placeholder(tf.float32, shape=[None, assets])
        self.input_future_prices = tf.placeholder(tf.float32, shape=[None, assets])
        self.out_nn = self._build_output_network()
        self.train_op = self._build_train_operation()

    def _build_output_network(self):
        layers_cnf = self.config["layers"]
        nn = tf.transpose(self.input_prices, [0, 2, 3, 1])
        for lc in layers_cnf[:-1]:
            nn = self._make_layer(nn, lc)
            self.layers.append(nn)

        nn = self._make_output_layer(nn, layers_cnf[-1])
        self.layers.append(nn)
        return nn

    @staticmethod
    def _make_layer(prev, config):
        return tflearn.layers.conv_2d(prev, config["out_channels"], config["kernel"],
                                      config.get("strides", DEFAULT_STRIDES),
                                      config.get("padding", DEFAULT_PADDING),
                                      config.get("activation", DEFAULT_ACTIVATION),
                                      regularizer=config.get("regularizer", None),
                                      weight_decay=config.get("weight_decay", DEFAULT_WEIGHT_DECAY))

    def _make_output_layer(self, prev, config):
        prev = self._add_previous_omega(prev)
        prev = self._make_layer(prev, config)
        prev = prev[:, :, 0, 0]
        prev = self._attach_cash_bias(prev)
        return tflearn.layers.activation(prev, activation="softmax")

    def _add_previous_omega(self, nn):
        shape = nn.shape
        nn = tf.reshape(nn, [-1, shape[1], 1, shape[2] * shape[3]])
        w = tf.reshape(self.input_prev_omega, [-1, shape[1], 1, 1])
        nn = tf.concat([nn, w], axis=3)
        return nn

    @staticmethod
    def _attach_cash_bias(nn):
        cash_bias = tf.get_variable("cash_bias", [1, 1], dtype=tf.float32, initializer=tf.zeros_initializer)
        cash_bias = tf.tile(cash_bias, [tf.shape(nn)[0], 1])
        nn = tf.concat([cash_bias, nn], axis=1)
        return nn

    def _build_train_operation(self):
        future_prices = self._add_cash(self.input_future_prices)
        mu = self._calc_commission(future_prices, self.out_nn)
        portfolio_values = tf.reduce_sum(future_prices * self.out_nn, reduction_indices=[1]) * \
                           tf.concat([tf.ones(1), mu], axis=0)
        loss = -tf.reduce_mean(tf.log(portfolio_values))
        train_cnf = self.config["training"]
        lr = tf.train.exponential_decay(train_cnf["learning_rate"], self.global_step, train_cnf["decay_steps"],
                                        train_cnf["decay_rate"],
                                        staircase=True)
        return tf.train.AdadeltaOptimizer(lr).minimize(loss, global_step=self.global_step)

    @staticmethod
    def _add_cash(relative_prices):
        return tf.concat([tf.ones([tf.shape(relative_prices)[0], 1]), relative_prices], axis=1)

    def _calc_commission(self, future_prices, omega):
        future_omega = self._calc_future_omega(future_prices, omega)
        w_prime = future_omega[:-1]
        w = omega[1:]
        return 1 - tf.reduce_sum(tf.abs(w_prime[:, 1:] - w[:, 1:]), axis=1) * self.config["commission"]

    @staticmethod
    def _calc_future_omega(future_prices, omega):
        # w' = y * w / |y . w|
        return (future_prices * omega) / (tf.reduce_sum(future_prices * omega, axis=1)[:, None])

    def predict(self, prices, previous_omega):
        res = self.session.run(self.out_nn, feed_dict={self.input_prices: np.expand_dims(prices, axis=0),
                                                       self.input_prev_omega: np.expand_dims(previous_omega, axis=0)})
        return res[0]

    def train(self, states, future_prices):
        prices, omegas = states
        result = self.session.run([self.train_op, self.out_nn], feed_dict={self.input_prices: prices,
                                                                           self.input_prev_omega: omegas,
                                                                           self.input_future_prices: future_prices})
        return result[-1]
