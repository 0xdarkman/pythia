import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns, hidden_units=[1])

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array([[0], [1], [2], [3]])},
    y=np.array([[0], [1], [2], [3]]),
    shuffle=False,
    num_epochs=None
)

regressor.train(input_fn=train_input_fn, steps=10000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array([[0], [2]])},
    y=np.array([[0], [2]]),
    shuffle=False,
)

score = regressor.evaluate(input_fn=test_input_fn)
print(score)

print("\n")

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array([[1]])},
    shuffle=False
)

predictions = regressor.predict(input_fn=predict_input_fn)
for pred in predictions:
    print(pred)
    print("\n")
    print(pred["predictions"])
    print("\n")
    print(pred["predictions"][0])
