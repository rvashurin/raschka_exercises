import tensorflow as tf
import tensorflow.keras as k

model = k.Sequential()
model.add(k.layers.Dense(units=16, activation='relu'))
model.add(k.layers.Dense(units=32, activation='relu'))

model.build(input_shape=(None, 4))
model.summary()

## printing variables of the model
for v in model.variables:
    print('{:20s}'.format(v.name), v.trainable, v.shape)

model = k.Sequential()
model.add(
    k.layers.Dense(
        units=16,
        activation=k.activations.relu,
        kernel_initializer=k.initializers.glorot_uniform(),
        bias_initializer=k.initializers.Constant(2.0)
    ))
model.add(
    k.layers.Dense(
        units=32,
        activation=k.activations.sigmoid,
        kernel_regularizer=k.regularizers.l1
    ))

model.compile(
    optimizer=k.optimizers.SGD(learning_rate=0.001),
    loss=k.losses.BinaryCrossentropy(),
    metrics=[k.metrics.Accuracy(),
             k.metrics.Precision(),
             k.metrics.Recall()]
    )

# Learning XOR:
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(1)
np.random.seed(1)

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

fig = plt.figure(figsize=(6, 6))
plt.plot(x[y==0, 0],
         x[y==0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y==1, 0],
         x[y==1, 1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'%x_1$', size=15)
plt.ylabel(r'%x_2$', size=15)
plt.show()

# Let's try simple model
model = k.Sequential()
model.add(k.layers.Dense(units=1, input_shape=(2,), activation='sigmoid'))
model.summary()

model.compile(optimizer=k.optimizers.SGD(),
              loss=k.losses.BinaryCrossentropy(),
              metrics=k.metrics.BinaryAccuracy())

hist = model.fit(x_train, y_train,
                 validation_data=(x_valid, y_valid),
                 epochs=200, batch_size=2, verbose=0)

from mlxtend.plotting import plot_decision_regions

history = hist.history

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer), clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()

tf.random.set_seed(1)
model = k.Sequential()
model.add(k.layers.Dense(units=4, input_shape=(2,), activation='relu'))
model.add(k.layers.Dense(units=4, activation='relu'))
model.add(k.layers.Dense(units=4, activation='relu'))
model.add(k.layers.Dense(units=1, activation='sigmoid'))

model.summary()

## compile
model.compile(optimizer=k.optimizers.SGD(),
              loss=k.losses.BinaryCrossentropy(),
              metrics=[k.metrics.BinaryAccuracy()])

## train
hist = model.fit(x_train, y_train,
                 validation_data=(x_valid, y_valid),
                 epochs=200, batch_size=2, verbose=0)

history = hist.history

fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer), clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()
