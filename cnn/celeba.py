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
