import tensorflow as tf

ACTIVATIONS = {
    'linear': tf.identity,
    'relu': tf.nn.relu
}


def _activation_func(activation):
    func = ACTIVATIONS.get(activation)
    if func is None:
        raise NotImplementedError("The specified activation function '{}' has not been implemented".format(activation))
    return func


LOSSES = {
    'mean_squared_error': tf.losses.mean_squared_error
}


def _loss_func(loss):
    func = LOSSES.get(loss)
    if func is None:
        raise NotImplementedError("The specified loss function '{}' has not been implemented".format(loss))
    return func


OPTIMIZERS = {
    'gradient_decent': tf.train.GradientDescentOptimizer
}


def _optimizer(optimizer):
    cls = OPTIMIZERS.get(optimizer)
    if cls is None:
        raise NotImplementedError("The specified optimizer '{}' has not been implemented".format(optimizer))
    return cls


class NeuralNetwork:
    def __init__(self, input_size, seed=None):
        self.inputs = tf.placeholder(tf.float32, [None, input_size])
        self.last_layer = self.inputs
        self.last_units = input_size
        self.layer_idx = 0
        self.targets = None
        self.train_step = None

        self.initializers = {
            'ones': tf.ones_initializer(),
            'zeros': tf.zeros_initializer(),
            'glorot_uniform': tf.glorot_uniform_initializer(seed)
        }

    def add_layer(self, units, activation, weight_init, bias_init='zeros'):
        if self.train_step is not None:
            raise InvalidOperationError("Adding layers after compiling a network is not supported")
        W = self._make_weights(weight_init, units)
        b = self._make_bias(bias_init, units)
        self.last_layer = _activation_func(activation)(tf.matmul(self.last_layer, W) + b)
        self.last_units = units
        self.layer_idx += 1

    def _make_weights(self, init, units):
        return tf.get_variable("W{}".format(self.layer_idx), shape=[self.last_units, units],
                               initializer=self._initializer(init))

    def _initializer(self, init):
        func = self.initializers.get(init)
        if func is None:
            raise NotImplementedError("The specified initialization '{}' has not been implemented".format(init))
        return func

    def _make_bias(self, bias_init, units):
        return tf.get_variable("b{}".format(self.layer_idx), shape=[units], initializer=self._initializer(bias_init))

    def compile(self, loss, optimizer, learning_rate):
        if self.layer_idx == 0:
            raise InvalidOperationError("A Neural Network needs at least one layer to be compiled")
        self.targets = tf.placeholder(tf.float32, shape=[None, self.last_units])
        loss_func = _loss_func(loss)(self.targets, self.last_layer)
        self.train_step = _optimizer(optimizer)(learning_rate).minimize(loss_func)

        tf.global_variables_initializer().run()

    def predict(self, inputs):
        return self.last_layer.eval(feed_dict={self.inputs: inputs})

    def train(self, inputs, targets):
        if self.train_step is None:
            raise NotCompiledError("The network needs to be compiled before it can be trained")
        self.train_step.run(feed_dict={self.inputs: inputs, self.targets: targets})


class NotCompiledError(Exception):
    pass


class InvalidOperationError(Exception):
    pass