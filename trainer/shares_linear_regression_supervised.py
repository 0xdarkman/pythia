import math

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.data import Dataset
from sklearn import metrics

WINDOW = 100


def calc_normalized_diffs(stock_frame):
    stocks = stock_frame['close'].dropna()
    diffs = stocks.shift(-1).dropna().truediv(stocks[:-1].dropna())
    return diffs


def pre_process_features(stock_frame):
    diffs = calc_normalized_diffs(stock_frame)[:-1]
    return pd.DataFrame({"close_{}".format(i): diffs.shift(-i) for i in range(WINDOW)}).dropna()


def pre_process_targets(stock_frame):
    diffs = calc_normalized_diffs(stock_frame)
    return diffs.shift(-WINDOW).dropna()


def construct_feature_columns(input_features):
    columns = set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])
    return columns


def input_fn(features, targets, batch_size=1, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    f, l = ds.make_one_shot_iterator().get_next()
    return f, l


def train_model(lr,
                steps,
                batch_size,
                training_examples,
                training_targets,
                validation_examples,
                validation_targets):
    periods = 10
    steps_per_period = steps / periods
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=optimizer
    )

    training_input_fn = lambda: input_fn(training_examples,
                                         training_targets,
                                         batch_size=batch_size)
    predict_training_input_fn = lambda: input_fn(training_examples,
                                                 training_targets,
                                                 num_epochs=1)
    predict_validation_input_fn = lambda: input_fn(validation_examples,
                                                   validation_targets,
                                                   num_epochs=1)

    print("Training the model...")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        linear_regressor.train(input_fn=training_input_fn,
                                steps=steps_per_period)

        training_preds = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_preds = np.array([item['predictions'][0] for item in training_preds])

        validation_preds = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_preds = np.array([item['predictions'][0] for item in validation_preds])

        t_rmse = math.sqrt(metrics.mean_squared_error(training_preds, training_targets))
        v_rmse = math.sqrt(metrics.mean_squared_error(validation_preds, validation_targets))
        print(" period {}: {}".format(period, t_rmse))
        training_rmse.append(t_rmse)
        validation_rmse.append(v_rmse)

    print("Model training finished")
    validation_preds = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_preds = np.array([item['predictions'][0] for item in validation_preds])
    print(math.fsum(validation_preds - validation_targets) / len(validation_targets))

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("RMSE vs Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor


stock_frame = pd.read_csv("../data/recordings/shares/SPY.csv")

stock_frame.close.plot()
plt.show()

print(stock_frame.describe())

total = len(stock_frame)
train_part = int(total * 0.8)
validation_part = total - train_part
training_examples = pre_process_features(stock_frame.head(train_part))
training_targets = pre_process_targets(stock_frame.head(train_part))
validation_examples = pre_process_features(stock_frame.tail(validation_part))
validation_targets = pre_process_targets(stock_frame.tail(validation_part))

_ = train_model(
    lr=0.0001,
    steps=1000,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
