import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import os

os.environ['NO_GCE_CHECK'] = 'true'

celeba_bldr = tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()

datasets = celeba_bldr.as_dataset(shuffle_files=False)
print(datasets.keys())

ds_train = datasets['train']
assert isinstance(ds_train, tf.data.Dataset)
example = next(iter(ds_train))
print(type(example))
print(example.keys())

# reformat dataset
ds_train = ds_train.map(lambda item: (item['image'],
                                      tf.cast(item['attributes']['Male'], tf.int32)))

ds_train = ds_train.batch(18)
images, labels = next(iter(ds_train))
print(images.shape, labels)

fig = plt.figure(figsize=(12, 8))
for i, (image, label) in enumerate(zip(images, labels)):
    ax = fig.add_subplot(3, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    ax.set_title('{}'.format(label), size=15)
plt.show()

# Alternative way to obtain datasets from tfds
mnist, mnist_info = tfds.load('mnist', with_info=True, shuffle_files=False)
print(mnist_info)

ds_train = mnist['train']
ds_train = ds_train.map(lambda item: (item['image'], item['label']))
ds_train = ds_train.batch(10)
batch = next(iter(ds_train))
print(batch[0].shape, batch[1])

fig = plt.figure(figsize=(15, 6))
for i, (image, label) in enumerate(zip(batch[0], batch[1])):
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image[:, :, 0], cmap='gray_r')
    ax.set_title('{}'.format(label), size=15)
plt.show()
