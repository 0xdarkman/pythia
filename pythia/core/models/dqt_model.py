from keras import Sequential
from keras import backend as K
from keras.layers import Dense
from keras.optimizers import Adam


class DQTModel:
    def __init__(self, input_size, lr, hidden_layers, seed=None):
        self.ann = Sequential()
        self.ann.add(Dense(hidden_layers[0], activation='relu', input_dim=input_size))
        for units in hidden_layers[1:]:
            self.ann.add(Dense(units, activation='relu'))
        self.ann.add(Dense(3, activation='linear'))
        self.ann.compile(optimizer=Adam(lr=lr), loss='mse')

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def train(self, batch):
        self.ann.fit(batch["inputs"], batch["targets"], steps_per_epoch=1, epochs=1, verbose=0)

    def predict(self, input):
        input = input.reshape([1, len(input)])
        return self.ann.predict(input, steps=1, verbose=0)[0]

    def interpolate(self, model, factor):
        mine = self.ann.get_weights()
        other = model.ann.get_weights()
        for i, o in enumerate(other):
            mine[i] = o * factor + (1 - factor) * mine[i]

        self.ann.set_weights(mine)

