import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow.keras as k
import tensorflow as tf
import numpy as np

celeba_bldr = tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_files=False)

celeba_train = celeba['train']
celeba_valid = celeba['validation']
celeba_test = celeba['test']

# too much - let's take only part
celeba_train = celeba_train.take(16000)
celeba_valid = celeba_valid.take(1000)

###### Data Augmentation
# take 5 examples
examples = []
for example in celeba_train.take(5):
    examples.append(example['image'])

fig = plt.figure(figsize=(16, 8.5))

## Column 1: cropping to a bounding-box
ax = fig.add_subplot(2, 5, 1)
ax.set_title('Crop to a \nbounding box', size=15)
ax.imshow(examples[0])
ax = fig.add_subplot(2, 5, 6)
img_cropped = tf.image.crop_to_bounding_box(
    examples[0], 50, 20, 128, 128)
ax.imshow(img_cropped)

## Column 2: Flipping horizontally
ax = fig.add_subplot(2, 5, 2)
ax.set_title('Flip', size=15)
ax.imshow(examples[1])
ax = fig.add_subplot(2, 5, 7)
img_flipped = tf.image.flip_left_right(examples[1])
ax.imshow(img_flipped)

## Column 3: adjust contrast
ax = fig.add_subplot(2, 5, 3)
ax.set_title('Adjust contrast', size=15)
ax.imshow(examples[2])
ax = fig.add_subplot(2, 5, 8)
img_adj_contrast = tf.image.adjust_contrast(examples[2], contrast_factor=2)
ax.imshow(img_adj_contrast)

## Column 4: adjust_brightness
ax = fig.add_subplot(2, 5, 4)
ax.set_title('Adjust brightness', size=15)
ax.imshow(examples[3])
ax = fig.add_subplot(2, 5, 9)
img_adj_brightness = tf.image.adjust_brightness(examples[2], delta=0.3)
ax.imshow(img_adj_brightness)

## Column 5: cropping from image center
ax = fig.add_subplot(2, 5, 5)
ax.set_title('Central crop\nand resize', size=15)
ax.imshow(examples[4])
ax = fig.add_subplot(2, 5, 10)
img_center_crop = tf.image.central_crop(examples[4], 0.7)
img_resized = tf.image.resize(img_center_crop, size=(218, 178))
ax.imshow(img_resized.numpy().astype('uint8'))

plt.show()

# Now all together and with random params:
tf.random.set_seed(1)

fig = plt.figure(figsize=(14, 12))

for i, example in enumerate(celeba_train.take(3)):
    image = example['image']

    ax = fig.add_subplot(3, 4, i * 4 + 1)
    ax.imshow(image)
    if i == 0:
        ax.set_title('Orig', size=15)

    ax = fig.add_subplot(3, 4, i * 4 + 2)
    img_crop = tf.image.random_crop(image, size=(178, 178, 3))
    ax.imshow(img_crop)
    if i == 0:
        ax.set_title('Step 1: Random crop', size=15)

    ax = fig.add_subplot(3, 4, i * 4 + 3)
    img_flip = tf.image.random_flip_left_right(img_crop)
    ax.imshow(tf.cast(img_flip, tf.uint8))
    if i == 0:
        ax.set_title('Step 2: Random flip', size=15)

    ax = fig.add_subplot(3, 4, i * 4 + 4)
    img_resize = tf.image.resize(img_flip, size=(128, 128))
    ax.imshow(tf.cast(img_resize, tf.uint8))
    if i == 0:
        ax.set_title('Step 3: Resize', size=15)

plt.show()

# Now combine it into pipeline
def preprocess(example, size=(64, 64), mode='train'):
    image = example['image']
    label = example['attributes']['Male']
    if mode == 'train':
        image_cropped = tf.image.random_crop(
            image, size=(178, 178, 3))
        image_resized = tf.image.resize(
            image_cropped, size=size)
        image_flip = tf.image.random_flip_left_right(
            image_resized)
        return image_flip/255.0, tf.cast(label, tf.int32)
    else: # use center- instead of random-crops for non-training data
        image_cropped = tf.image.crop_to_bounding_box(
            image, offset_height=20, offset_width=0,
            target_height=178, target_width=178)
        image_resized = tf.image.resize(
            image_cropped, size=size)
        return image_resized/255.0, tf.cast(label, tf.int32)

tf.random.set_seed(1)

ds = celeba_train.shuffle(1000, reshuffle_each_iteration=False)
ds = ds.take(2).repeat(5)

ds = ds.map(lambda x:preprocess(x, size=(178, 178), mode='train'))

fig = plt.figure(figsize=(15, 6))
for j, example in enumerate(ds):
    ax = fig.add_subplot(2, 5, j//2+(j%2)*5+1)  ## WTF hahaha
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])
plt.show()

## Apply this pipeline to Celeba data
BATCH_SIZE = 32
BUFFER_SIZE = 1000
IMAGE_SIZE = (64, 64)
steps_per_epoch = np.ceil(16000/BATCH_SIZE)

ds_train = celeba_train.map(
    lambda x: preprocess(x, size=IMAGE_SIZE, mode='train'))
ds_train = ds_train.shuffle(buffer_size=BUFFER_SIZE).repeat()
ds_train = ds_train.batch(BATCH_SIZE)

ds_valid = celeba_valid.map(
    lambda x: preprocess(x, size=IMAGE_SIZE, mode='eval'))
ds_valid = ds_valid.batch(BATCH_SIZE)

model = k.Sequential([
    k.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu'),
    k.layers.MaxPooling2D((2, 2)),
    k.layers.Dropout(rate=0.5),

    k.layers.Conv2D(
        64, (3, 3), padding='same', activation='relu'),
    k.layers.MaxPooling2D((2, 2)),
    k.layers.Dropout(rate=0.5),

    k.layers.Conv2D(
        128, (3, 3), padding='same', activation='relu'),
    k.layers.MaxPooling2D((2, 2)),

    k.layers.Conv2D(
        256, (3, 3), padding='same', activation='relu')
])

model.compute_output_shape(input_shape=(None, 64, 64, 3))

model.add(k.layers.GlobalAveragePooling2D())
model.compute_output_shape(input_shape=(None, 64, 64, 3))
model.add(k.layers.Dense(1, activation=None))

tf.random.set_seed(1)

model.build(input_shape=(None, 64, 64, 3))
model.summary()

model.compile(optimizer=k.optimizers.Adam(),
              loss=k.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(ds_train, validation_data=ds_valid,
                    epochs=20, steps_per_epoch=steps_per_epoch)

hist = history.history

x_arr = np.arange(len(hist['loss'])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.show()

# No plateau:
history = model.fit(ds_train, validation_data=ds_valid, epochs=30,
                    initial_epoch=20, steps_per_epoch=steps_per_epoch)

hist = history.history

x_arr = np.arange(len(hist['loss'])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.show()

ds_test = celeba_test.map(
    lambda x: preprocess(x, size=IMAGE_SIZE, mode='eval')).batch(32)

test_results = model.evaluate(ds_test)
print('Test Acc: {:.2f}%'.format(test_results[1]*100))

ds = ds_test.unbatch().take(10)

pred_logits = model.predict(ds.batch(10))
probas = tf.sigmoid(pred_logits)
probas = probas.numpy().flatten()*100

fig = plt.figure(figsize=(15, 7))
for j, example in enumerate(ds):
    ax = fig.add_subplot(2, 5, j+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0])
    if example[1].numpy() == 1:
        label='M'
    else:
        label='F'
    ax.text(
        0.5, -0.15, 'GT: {:s}\nPr(Male)={:.0f}%'
        ''.format(label, probas[j]),
        size=16,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)
plt.tight_layout()
plt.show()
