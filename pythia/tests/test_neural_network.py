import tensorflow as tf

from pythia.core.reinforcement.neural_network import NeuralNetwork, NotCompiledError, InvalidOperationError


class AnnBuilder:
    def __init__(self, input_size, seed):
        self.ann = NeuralNetwork(input_size, seed)
        self.loss = 'mean_squared_error'
        self.optimizer = 'gradient_descent'
        self.lr = 0.1

    def add_layer(self, units, activation, weight_init, bias_init='zeros'):
        self.ann.add_layer(units, activation, weight_init, bias_init)
        return self

    def compile(self, loss, optimizer, lr):
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        return self

    def finish(self):
        self.ann.compile(self.loss, self.optimizer, self.lr)
        return self.ann


def make_ann(input_size, seed=None):
    return AnnBuilder(input_size, seed)


class NeuralNetworkTest(tf.test.TestCase):
    def setUp(self):
        self.original_v = tf.logging.get_verbosity()
        tf.logging.set_verbosity(3)
        tf.set_random_seed(42)
        tf.reset_default_graph()

    def tearDown(self):
        tf.logging.set_verbosity(self.original_v)
        tf.set_random_seed(None)

    def test_zero_layers(self):
        with self.assertRaises(InvalidOperationError) as e_info, self.test_session():
            make_ann(input_size=1).finish()
        self.assertEqual(str(e_info.exception), "A Neural Network needs at least one layer to be compiled")

    def test_simple_two_neuron_network(self):
        with self.test_session():
            ann = make_ann(input_size=1).add_layer(1, 'linear', 'ones').finish()
            self.assertAllEqual(ann.predict([[3]]), [[3]])
            self.assertAllEqual(ann.predict([[-1]]), [[-1]])

    def test_relu_activation(self):
        with self.test_session():
            ann = make_ann(input_size=1).add_layer(1, 'relu', 'ones').finish()
            self.assertAllEqual(ann.predict([[3]]), [[3]])
            self.assertAllEqual(ann.predict([[-1]]), [[0]])

    def test_missing_activation_function(self):
        with self.assertRaises(NotImplementedError) as e_info, self.test_session():
            make_ann(input_size=1).add_layer(1, 'MISSING', 'ones').finish()
        self.assertEqual(str(e_info.exception), "The specified activation function 'MISSING' has not been implemented")

    def test_weighted_inputs(self):
        with self.test_session():
            ann = make_ann(input_size=2).add_layer(1, 'linear', 'ones').finish()
            self.assertAllEqual(ann.predict([[3, 2]]), [[5]])

    def test_zero_initialization(self):
        with self.test_session():
            ann = make_ann(input_size=2).add_layer(1, 'linear', 'zeros').finish()
            self.assertAllEqual(ann.predict([[3, 2]]), [[0]])

    def test_missing_initialization(self):
        with self.assertRaises(NotImplementedError) as e_info, self.test_session():
            make_ann(input_size=2).add_layer(1, 'linear', 'MISSING').finish()
        self.assertEqual(str(e_info.exception), "The specified initialization 'MISSING' has not been implemented")

    def test_glorot_uniform_initialization(self):
        with self.test_session():
            ann = make_ann(input_size=1, seed=7).add_layer(1, 'linear', 'glorot_uniform').finish()
            self.assertAllClose(ann.predict([[3]]), [[-0.16689062]])

    def test_bias_initialization(self):
        with self.test_session():
            ann = make_ann(input_size=1).add_layer(1, 'linear', 'zeros', bias_init='ones').finish()
            self.assertAllEqual(ann.predict([[3]]), [[1]])

    def test_glorot_uniform_bias_initialization(self):
        with self.test_session():
            ann = make_ann(input_size=1, seed=7).add_layer(1, 'linear', 'zeros', bias_init='glorot_uniform').finish()
            self.assertAllClose(ann.predict([[3]]), [[-0.05563021]])

    def test_missing_bias_initialization(self):
        with self.assertRaises(NotImplementedError) as e_info, self.test_session():
            make_ann(input_size=2).add_layer(1, 'linear', 'zeros', bias_init='MISSING').finish()
        self.assertEqual(str(e_info.exception), "The specified initialization 'MISSING' has not been implemented")

    def test_training_uncompiled_network(self):
        with self.assertRaises(NotCompiledError) as e_info, self.test_session():
            ann = NeuralNetwork(input_size=2)
            ann.add_layer(1, 'linear', 'zeros')
            ann.train([[3, 1], [1, 2]], [[-1], [2]])
        self.assertEqual(str(e_info.exception), "The network needs to be compiled before it can be trained")

    def test_mean_squared_error_gradient_descent_training(self):
        with self.test_session():
            ann = make_ann(input_size=1).add_layer(1, 'linear', 'zeros') \
                .compile('mean_squared_error', 'gradient_descent', lr=0.5).finish()
            ann.train([[0]], [[10]])
            self.assertAllClose(ann.predict([[0]]), [[10]])

    def test_different_learning_rate(self):
        with self.test_session():
            ann = make_ann(input_size=1).add_layer(1, 'linear', 'zeros') \
                .compile('mean_squared_error', 'gradient_descent', lr=0.1).finish()
            ann.train([[0]], [[10]])
            self.assertAllClose(ann.predict([[0]]), [[2]])

    def test_missing_loss_function(self):
        with self.assertRaises(NotImplementedError) as e_info, self.test_session():
            make_ann(input_size=1).add_layer(1, 'linear', 'zeros') \
                .compile('MISSING', 'gradient_descent', lr=0.1).finish()
        self.assertEqual(str(e_info.exception), "The specified loss function 'MISSING' has not been implemented")

    def test_missing_optimizer(self):
        with self.assertRaises(NotImplementedError) as e_info, self.test_session():
            make_ann(input_size=1).add_layer(1, 'linear', 'zeros') \
                .compile('mean_squared_error', 'MISSING', lr=0.1).finish()
        self.assertEqual(str(e_info.exception), "The specified optimizer 'MISSING' has not been implemented")

    def test_adding_layer_after_compilation(self):
        with self.assertRaises(InvalidOperationError) as e_info, self.test_session():
            ann = NeuralNetwork(input_size=1)
            ann.add_layer(1, 'linear', 'zeros')
            ann.compile('mean_squared_error', 'gradient_descent', 0.1)
            ann.add_layer(1, 'linear', 'zeros')
        self.assertEqual(str(e_info.exception), "Adding layers after compiling a network is not supported")

    def test_deep_neural_network_prediction(self):
        with self.test_session():
            ann = make_ann(input_size=2).add_layer(2, 'relu', 'ones').add_layer(1, 'linear', 'ones').finish()
            self.assertAllEqual(ann.predict([[3, 2]]), [[10]])
            self.assertAllEqual(ann.predict([[3, -1]]), [[4]])

    def test_deep_neural_network_training(self):
        with self.test_session():
            ann = make_ann(input_size=1).add_layer(1, 'linear', 'zeros').add_layer(1, 'linear', 'zeros') \
                .compile('mean_squared_error', 'gradient_descent', lr=0.5).finish()
            ann.train([[0]], [[10]])
            self.assertAllClose(ann.predict([[0]]), [[10]])


if __name__ == '__main__':
    tf.test.main()
