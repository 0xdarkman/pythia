import random

from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.models import Sequential

from pythia.reinforcement.q_model import QModel


class QRegressionKerasModel(QModel):
    def __init__(self, input_size, hidden_layers):
        QModel.__init__(self, input_size)
        self.input_size = input_size
        self.regressor = Sequential()
        self.regressor.add(Dense(hidden_layers[0], input_dim=input_size, activation="relu"))
        for i in range(1, len(hidden_layers)):
            self.regressor.add(Dense(hidden_layers[i], activation="relu"))
        self.regressor.add(Dense(1, activation="linear"))
        self.regressor.compile("sgd", "mean_squared_error")
        self.is_untrained = True

    def do_prediction(self, state):
        if self.is_untrained:
            return [2 * random.random() - 1]
        predictions = self.regressor.predict(state)
        return predictions

    def do_training(self, states, targets):
        self.regressor.fit(states, targets, batch_size=len(states), epochs=1)
        self.is_untrained = False
