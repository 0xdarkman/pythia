import numpy as np

from reinforcement.models.q_regression_model import QRegressionModel

model = QRegressionModel(1, [], 0.01)

for _ in range(0, 100):
    model.train(np.array([[2]]), np.array([[2]]))

prediction = model.predict(np.array([[2]]))

print(prediction)
