import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from pythia.core.environment.simulators import PsychicTrader
from pythia.core.environment.stocks import StockData, StockVisualizer

TEST_SIZE = 100
LOOK_BACK = 60

stocks = StockData("model_data/sxe.csv")
test_trader = PsychicTrader(1000.0, stocks.data[:, 0], 0.01)

scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(stocks.data)

encoder = OneHotEncoder()
encoded_actions = encoder.fit_transform(np.array(test_trader.actions).reshape(-1, 1)).toarray()

X_train = []
y_train = []
for i in range(LOOK_BACK, len(training_set_scaled) - TEST_SIZE):
    X_train.append(training_set_scaled[i - LOOK_BACK:i])
    y_train.append(encoded_actions[i])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

classifier = Sequential()
classifier.add(LSTM(units=512, activation='relu', input_shape=(None, 5), dropout=0.9, return_sequences=True))
classifier.add(LSTM(units=256, activation='relu', dropout=0.8))
classifier.add(Dense(units=3, activation='softmax'))

sgd = optimizers.SGD(lr=0.1, decay=1e-16)
classifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

classifier.fit(X_train, y_train, epochs=3600, batch_size=32)

X_test = []
y_test = []
for i in range((len(training_set_scaled) - TEST_SIZE), len(training_set_scaled)):
    X_test.append(training_set_scaled[i - LOOK_BACK:i])
    y_test.append(encoded_actions[i])

X_test, y_test = np.array(X_test), np.array(y_test)
X_train = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

pred_actions = []
for code in y_pred:
    if code[0]:
        pred_actions.append(0)
    elif code[1]:
        pred_actions.append(1)
    else:
        pred_actions.append(2)

print(test_trader.portfolio)
view = StockVisualizer(stocks.data[(len(training_set_scaled) - TEST_SIZE):], pred_actions)
view.plot()
