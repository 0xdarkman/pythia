import tensorflow as tf

from pythia.reinforcement.q_model import QModel


class QRegressionModel(QModel):
    def __init__(self, input_size, hidden_layers, lr=0.01):
        QModel.__init__(self, input_size)
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, input_size])
        layer = self.x
        prev_size = input_size
        for i, layer_size in enumerate(hidden_layers):
            W = tf.get_variable("W{}".format(i), shape=[prev_size, layer_size], initializer=tf.glorot_uniform_initializer())
            b = tf.Variable(tf.zeros([layer_size]))
            layer = tf.nn.relu(tf.matmul(layer, W) + b)
            prev_size = layer_size

        W_o = tf.get_variable("W_o", shape=[prev_size, 1], initializer=tf.glorot_uniform_initializer())
        b_o = tf.Variable(tf.zeros([1]))
        self.y = tf.matmul(layer, W_o) + b_o

        self.y_ = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.losses.mean_squared_error(self.y_, self.y)
        self.train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def do_prediction(self, state):
        return self.sess.run(self.y, feed_dict={self.x: state})[0]

    def do_training(self, states, targets):
        self.sess.run(self.train_step, feed_dict={self.x: states, self.y_: targets})
