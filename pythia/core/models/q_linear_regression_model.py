import tensorflow as tf

from reinforcement.models.q_model import QModel


class QLinearRegressionModel(QModel):
    def __init__(self, input_size, lr):
        super().__init__(input_size)

        self._inputs = tf.placeholder(tf.float32, [None, input_size])
        W = tf.get_variable("W_Features", shape=[input_size, 1])
        b = tf.get_variable("Bias", shape=[1])
        self._predicting = tf.matmul(self._inputs, W) + b

        self._targets = tf.placeholder(tf.float32, [None, 1])
        loss = tf.losses.mean_squared_error(self._targets, self._predicting)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
        self._train_step = optimizer.minimize(loss)

        tf.global_variables_initializer().run()

    def do_training(self, states, targets):
        _ = self._train_step.run(feed_dict={self._inputs: states, self._targets: targets})

    def do_prediction(self, state):
        return self._predicting.eval(feed_dict={self._inputs: state})
