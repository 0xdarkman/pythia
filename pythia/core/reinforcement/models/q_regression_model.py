from pythia.core.reinforcement.models.neural_network import NeuralNetwork
from .q_model import QModel


class QRegressionModel(QModel):
    def __init__(self, input_size, hidden_layers, lr=0.01, seed=None):
        """
        Implements a regression Artificial Neural Network using linear rectifier units as hidden neurons and outputs a
        linear float value. Training is done using a gradient descent and the loss is calculated using the mean squared
        error. Training and target batches are expected as a list of lists.

        :param input_size: Size of the input state vector
        :param hidden_layers: List specifying number and size of hidden relu neuron layers, i.e.: ``[3, 2]`` specify two hidden layers of size 3 and 2
        :param lr: Learning rate parameter
        :param seed: Seed used by random operations
        """
        QModel.__init__(self, input_size)
        self.ann = NeuralNetwork(input_size, seed)
        for units in hidden_layers:
            self.ann.add_layer(units, activation='relu', weight_init='glorot_uniform')
        self.ann.add_layer(1, activation='linear', weight_init='glorot_uniform')
        self.ann.compile('mean_squared_error', 'gradient_descent', lr)

    def do_prediction(self, state):
        return self.ann.predict(state)

    def do_training(self, states, targets):
        self.ann.train(states, targets)
