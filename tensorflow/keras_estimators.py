import tensorflow.keras as k
import tensorflow as tf
import numpy as np

tf.random.set_seed(1)
np.random.seed(1)

## Create the data
x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

model = k.Sequential([
    k.layers.Input(shape=(2,), name='input-features'),
    k.layers.Dense(units=4, activation='relu'),
    k.layers.Dense(units=4, activation='relu'),
    k.layers.Dense(units=4, activation='relu'),
    k.layers.Dense(units=1, activation='sigmoid')
])

## Step 1: Define the input functions

def train_input_fn(x_train, y_train, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'input-features':x_train}, y_train.reshape(-1, 1)))
    # shuffle, repeat, and batch examples
    return dataset.shuffle(100).repeat().batch(batch_size)

def eval_input_fn(x_test, y_test=None, batch_size=8):
    if y_test is None:
        dataset = tf.data.Dataset.from_tensor_slices(
            {'input-features':x_test})
    else:
       dataset = tf.data.Dataset.from_tensor_slices(
            ({'input-features':x_test}, y_test.reshape(-1, 1)))
    return dataset.batch(batch_size)

## Step 2: Define the feature columns
features = [
    tf.feature_column.numeric_column(
        key='input-features', shape=(2,))
]

model.compile(optimizer=k.optimizers.SGD(),
              loss=k.losses.BinaryCrossentropy(),
              metrics=[k.metrics.BinaryAccuracy()])

my_estimator = k.estimator.model_to_estimator(
    keras_model=model,
    model_dir='models/estimator-for-XOR/')

## Step 4: Use the estimator
num_epochs = 200
batch_size = 2
steps_per_epoch = np.ceil(len(x_train) / batch_size)

my_estimator.train(
    input_fn = lambda: train_input_fn(x_train, y_train, batch_size),
    steps=num_epochs * steps_per_epoch)

my_estimator.evaluate(
    input_fn = lambda: eval_input_fn(x_valid, y_valid, batch_size),
    steps=num_epochs * steps_per_epoch)
