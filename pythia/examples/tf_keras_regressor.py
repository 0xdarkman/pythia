import numpy as np
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.models import Sequential

model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer="zeros", activation="relu"))
model.add(Dense(1, activation="linear", kernel_initializer="zeros"))
model.compile(optimizer="sgd", loss="mean_squared_error")

model.fit(np.array([[2]]), np.array([[2]]), batch_size=1, epochs=1)

prediction = model.predict(np.array([[2]]), batch_size=1)

print(prediction)
