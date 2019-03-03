import itertools
import os
import shutil

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
        self.checkpoint_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "test_data/test_save_model/point.ckpt")
        self.checkpoint_restore = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               "test_data/test_restore_model/point.ckpt")

    def tearDown(self):
        tf.set_random_seed(None)
        tf.logging.set_verbosity(self.original_v)
        tf.reset_default_graph()
        ckpt_path = os.path.dirname(self.checkpoint_save)
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)

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

    def _do_make_test_output_of_cnn(self, f, m, n):
        with self.test_session() as sess:
            nn = self._make_ensemble_for_setup(sess, m, n)
            sess.run(tf.global_variables_initializer())
            return nn.train((np.ones([5, f, m, n]), np.ones([5, m])), np.ones([5, m]))

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

    def test_save_model(self):
        with self.test_session() as sess:
            nn = self._make_ensemble_for_setup(sess, 1, 2)
            sess.run(tf.global_variables_initializer())
            nn.save(self.checkpoint_save)

        os.path.exists(os.path.dirname(self.checkpoint_save))

    def test_restore_model(self):
        conv0_w = """[[[[ 0.38449544 -0.10550028  0.20254272]
   [ 0.26774544  0.11936188  0.56204623]
   [-0.49919772  0.27477652 -0.4346569 ]]

  [[ 0.3909791  -0.11569393  0.07186073]
   [-0.5447465  -0.6805804   0.10508257]
   [ 0.41523463 -0.6563621   0.486252  ]]]]"""

        conv0_b = """[0. 0. 0.]"""

        conv1_w = """[[[[-0.6971295   0.39889598 -0.83452344 -0.32657504 -0.22658801
     0.0329442   0.3347504   0.83522606 -0.43380594 -0.39102077]
   [ 0.8894398   0.2897029  -0.19237447  0.56116056 -0.83244085
     0.51532674 -0.5999918   0.24874973 -0.5619366  -0.09490609]
   [ 0.89980197 -0.46960068  0.09130788 -0.34258533  0.06571674
    -0.742718   -0.8767636  -0.50736666  0.8575001  -0.8801789 ]]]]"""

        conv1_b = """[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]"""

        tf.set_random_seed(21)
        with self.test_session() as sess:
            nn = self._make_ensemble_for_setup(sess, 1, 2)
            sess.run(tf.global_variables_initializer())
            nn.restore(self.checkpoint_restore)
            assert str(nn.layers[0].W.eval()) == conv0_w
            assert str(nn.layers[0].b.eval()) == conv0_b
            assert str(nn.layers[1].W.eval()) == conv1_w
            assert str(nn.layers[1].b.eval()) == conv1_b
