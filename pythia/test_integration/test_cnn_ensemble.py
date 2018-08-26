import itertools

import numpy as np
import tensorflow as tf

from pythia.core.agents.cnn_ensamble import DEFAULT_ACTIVATION, CNNEnsemble


def layer_config(out_channels, kernel, strides=None, activation=None, padding=None, regularizer=None,
                 weight_decay=None):
    cnf = dict({"out_channels": out_channels, "kernel": kernel})
    if activation:
        cnf["activation"] = activation
    if strides:
        cnf["strides"] = strides
    if padding:
        cnf["padding"] = padding
    if regularizer:
        cnf["regularizer"] = regularizer
    if weight_decay:
        cnf["weight_decay"] = weight_decay

    return cnf


def config(layers, commission=0.0, lr=0.001, decay_steps=0, decay_rate=0.0):
    return {"layers": layers, "trading": {"commission": commission},
            "training": {"learning_rate": lr, "decay_steps": decay_steps, "decay_rate": decay_rate}}


def assert_layer(layer, activation, shape):
    assert activation in layer.name.lower()
    assert shape == layer.shape.as_list()


def assert_is_buying(omega):
    assert omega[0] <= 0.1
    assert omega[1] >= 0.9


def assert_is_selling(omega):
    assert omega[0] >= 0.1
    assert omega[1] <= 0.9


def window(seq, n):
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class TimeSeries:
    def __init__(self, description, batch_size, window_size):
        self.batch_size = batch_size
        self.window_size = window_size
        prices = self._extract_prices(description)
        self.length = len(prices) - 1
        self.x, self.y = self._make_x_and_y_from_prices(prices)
        self.prev_w = np.random.randn(self.x.shape[0], 1)
        self.prev_w[0] = [0]
        self.cursor = -1

    @staticmethod
    def _extract_prices(description):
        prices = []
        lines = description.split("\n")
        for col in zip(*lines[:-1]):
            prices.append(len(lines) - 1 - col.index("+"))

        return prices

    def _make_x_and_y_from_prices(self, prices):
        x = []
        y = []
        for w in window(prices, self.window_size + 1):
            lc = w[-2]
            closing = np.array(w[:-1]) / lc
            high = (np.array(w[:-1]) * 1.1) / lc
            low = (np.array(w[:-1]) * 0.9) / lc
            x.append([[closing, high, low]])
            y.append([w[-1] / lc])

        xa = np.array(x)
        return xa.transpose((0, 2, 1, 3)), np.array(y)

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        end = self.cursor + self.batch_size
        if end == self.length:
            raise StopIteration

        return (self.x[self.cursor:end], self.prev_w[self.cursor:end]), self.y[self.cursor:end]

    def reset(self):
        self.cursor = -1

    def get_complete(self):
        return (self.x, self.prev_w), self.y

    def update(self, new_omega):
        start = self.cursor + 1
        end = min(start + self.batch_size, self.x.shape[0])
        ln = end - start
        self.prev_w[start:end] = new_omega[:ln, 1:]


def make_series(description, batch_size, window_size):
    return TimeSeries(description, batch_size, window_size)


class CNNEnsembleTests(tf.test.TestCase):
    def setUp(self):
        self.original_v = tf.logging.get_verbosity()
        tf.logging.set_verbosity("ERROR")
        tf.set_random_seed(42)

    def tearDown(self):
        tf.set_random_seed(None)
        tf.logging.set_verbosity(self.original_v)
        tf.reset_default_graph()

    def test_that_it_is_created_with_correct_configuration(self):
        with self.test_session() as sess:
            nn = CNNEnsemble(sess, 11, 50, config([layer_config(2, [1, 2], activation="sigmoid"),
                                                   layer_config(10, [1, 49], regularizer="L2", weight_decay=5e-9),
                                                   layer_config(1, [1, 1], regularizer="L2", weight_decay=5e-8)]))
            assert len(nn.layers) == 3
            assert_layer(nn.layers[0], "sigmoid", [None, 11, 49, 2])
            assert_layer(nn.layers[1], DEFAULT_ACTIVATION, [None, 11, 1, 10])
            assert_layer(nn.layers[2], "softmax", [None, 12])

    def test_training_takes_correct_input_and_produces_output_with_the_correct_shape(self):
        self._do_test_input_output_shapes(3, 11, 50)

    def test_edge_cases_of_input_output_shapes(self):
        self._do_test_input_output_shapes(3, 1, 2)

    def _do_test_input_output_shapes(self, f, m, n):
        with self.test_session() as sess:
            nn = self._make_ensemble_for_setup(sess, m, n)
            sess.run(tf.global_variables_initializer())
            out = nn.train((np.ones([5, f, m, n]), np.ones([5, m])), np.ones([5, m]))
            assert (5, m + 1) == out.shape

    @staticmethod
    def _make_ensemble_for_setup(sess, m, n):
        return CNNEnsemble(sess, m, n, config([layer_config(3, [1, 2]),
                                               layer_config(10, [1, n - 1], regularizer="L2", weight_decay=5e-9),
                                               layer_config(1, [1, 1], regularizer="L2", weight_decay=5e-8)]))

    def test_cnn_ensemble_can_perform_a_trivial_training_circle_with_no_error(self):
        with self.test_session() as sess:
            nn = CNNEnsemble(sess, 1, 2, config([layer_config(2, [1, 2]),
                                                 layer_config(3, [1, 1], regularizer="L2", weight_decay=5e-9),
                                                 layer_config(1, [1, 1], regularizer="L2", weight_decay=5e-8)],
                                                commission=0.0025, lr=0.001, decay_steps=50000, decay_rate=1.0))

            sess.run(tf.global_variables_initializer())
            series = make_series("      ++  \n"
                                 "++  ++  ++\n"
                                 "  ++      \n"
                                 "0123456789", 2, 2)
            episodes = 2
            for _ in range(episodes):
                for s, f in series:
                    out = nn.train(s, f)
                    series.update(out)
                series.reset()

    def test_cnn_predict_produces_action_with_correct_shape(self):
        self._do_test_predict_shape(3, 11, 50)

    def test_edge_case_for_input_and_output_prediction_shapes(self):
        self._do_test_predict_shape(3, 1, 2)

    def _do_test_predict_shape(self, f, m, n):
        with self.test_session() as sess:
            nn = self._make_ensemble_for_setup(sess, m, n)
            sess.run(tf.global_variables_initializer())
            a = nn.predict(np.ones([f, m, n]), np.ones([m]))
            assert (m + 1,) == a.shape
