import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow.keras as k
import tensorflow as tf
import numpy as np

## Loading the data
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)
mnist_train_orig = datasets['train']
mnist_test_orig = datasets['test']

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20

mnist_train = mnist_train_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32)/255.0,
                  tf.cast(item['label'], tf.int32)))

mnist_test = mnist_test_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32)/255.0,
                  tf.cast(item['label'], tf.int32)))

tf.random.set_seed(1)
mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE,
                                  reshuffle_each_iteration=False)

mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)

model = k.Sequential()
model.add(k.layers.Conv2D(
    filters=32, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    data_format='channels_last',
    name='conv_1', activation='relu'))
model.add(k.layers.MaxPool2D(
    pool_size=(2, 2), name='pool_1'))

model.add(k.layers.Conv2D(
    filters=64, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    name='conv_2', activation='relu'))
model.add(k.layers.MaxPool2D(
    pool_size=(2, 2), name='pool_2'))

# What is the output shape?
model.compute_output_shape(input_shape=(16, 28, 28, 1))

# Flatten the output of convolutional part for the dense layer:
model.add(k.layers.Flatten())
model.compute_output_shape(input_shape=(16, 28, 28, 1))

model.add(k.layers.Dense(
    units=1024, name='fc_1',
    activation='relu'))

model.add(k.layers.Dropout(
    rate=0.5))

model.add(k.layers.Dense(
    units=10, name='fc_2',
    activation='softmax'))

tf.random.set_seed(1)
model.build(input_shape=(None, 28, 28, 1))
model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

history = model.fit(mnist_train, epochs=NUM_EPOCHS,
                    validation_data=mnist_valid,
                    shuffle=True)

hist = history.history
x_arr = np.arange(len(hist['loss'])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
plt.show()

test_results = model.evaluate(mnist_test.batch(20))
print('Test Acc.: {:.2f}\%'.format(test_results[1]*100))

batch_test = next(iter(mnist_test.batch(12)))

preds = model(batch_test[0])
tf.print(preds.shape)

preds = tf.argmax(preds, axis=1)
print(preds)

fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i + 1)
    ax.set_xticks([]); ax.set_yticks([])
    img = batch_test[0][i, :, :, 0]
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, '{}'.format(preds[i]),
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.show()
