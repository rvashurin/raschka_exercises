import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import tensorflow.keras as k

df = pd.read_csv('movie_data.csv', encoding='utf-8')

# Step 1: create a dataset
target = df.pop('sentiment')
ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))

## inspection:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])

tf.random.set_seed(1)
ds_raw = ds_raw.shuffle(50000, reshuffle_each_iteration=False)

ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

## Step 2: find unique tokens (words)
from collections import Counter

tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)

print('Vocab-size:', len(token_counts))

## Step 3: encoding unique tokens to integers
encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)
example_str = 'This is an example!'
print(encoder.encode(example_str))

## Step 3-A: define the function for transformation
def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

## Step 3-B: wrap the encode funtion to a TF Op.
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label],
                          Tout=(tf.int64, tf.int64))

ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)

# look at the shape of some examples:
tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print('Sequence length:', example[0].shape)

## Take small subset
ds_subset = ds_train.take(8)
for example in ds_subset:
    print('Individual size:', example[0].shape)

## Dividing the dataset into batches
ds_batched = ds_subset.padded_batch(4, padded_shapes=([-1], []))

for batch in ds_batched:
    print('Batch dimension:', batch[0].shape)

train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

## Embedding
from tensorflow.keras.layers import Embedding

model = k.Sequential()

model.add(Embedding(input_dim=100,
                    output_dim=6,
                    input_length=20,
                    name='embed-layer'))
model.summary()

## Example RNN model with embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.add(Dense(1))
model.summary()

# ## Build the actual model:
# from tensorflow.keras.layers import Bidirectional
# from tensorflow.keras.layers import LSTM
#
# embedding_dim = 20
# vocab_size = len(token_counts) + 2
#
# tf.random.set_seed(1)
#
# bi_lstm_model = Sequential([
#     Embedding(input_dim=vocab_size,
#               output_dim=embedding_dim,
#               name='embed-layer'),
#     Bidirectional(
#         LSTM(64, name='lstm-layer'),
#         name='bidir-lstm'),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
#
# bi_lstm_model.summary()
#
# ## compile and train:
# bi_lstm_model.compile(
#     optimizer=k.optimizers.Adam(1e-3),
#     loss=k.losses.BinaryCrossentropy(from_logits=False),
#     metrics=['accuracy'])
#
# history = bi_lstm_model.fit(
#     train_data,
#     validation_data=valid_data,
#     epochs=10)
#
# ## evaluate on the test data
# test_results = bi_lstm_model.evaluate(test_data)
# print('Test Acc.: {:.2f}%'.format(test_results[1]*100))

## Not very good - let's try reducing sequence length by taking 100 last tokens
def preprocess_datasets(
    ds_raw_train,
    ds_raw_valid,
    ds_raw_test,
    max_seq_length=None,
    batch_size=32):

    ## (Step 1 is already done)
    ## Step 2: find unique tokens
    tokenizer = tfds.deprecated.text.Tokenizer()
    token_counts = Counter()

    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        if max_seq_length is not None:
            tokens = tokens[-max_seq_length:]
        token_counts.update(tokens)

    print('Vocab-size:', len(token_counts))

    ## Step 3: encoding the texts
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)
    def encode(test_tensor, label):
        text = test_tensor.numpy()[0]
        encoded_text = encoder.encode(text)
        if max_seq_length is not None:
            encoded_text = encoded_text[-max_seq_length:]
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label],
                              Tout=(tf.int64, tf.int64))

    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid = ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)

    ## Step 4: batching the datasets
    train_data = ds_train.padded_batch(
        batch_size, padded_shapes=([-1],[]))
    valid_data = ds_valid.padded_batch(
        batch_size, padded_shapes=([-1],[]))
    test_data = ds_test.padded_batch(
        batch_size, padded_shapes=([-1],[]))

    return (train_data, valid_data, test_data, len(token_counts))

from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

def build_rnn_model(embedding_dim, vocab_size, recurrent_type='SimpleRNN',
                    n_recurrent_units=64, n_recurrent_layers=1,
                    bidirectional=True):

    tf.random.set_seed(1)

    # build the model
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embed-layer'
        )
    )

    for i in range(n_recurrent_layers):
        return_sequences = (i < n_recurrent_layers - 1)

        if recurrent_type == 'SimpleRNN':
            recurrent_layer = SimpleRNN(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='simprnn-layer-{}'.format(i))
        elif recurrent_type == 'LSTM':
            recurrent_layer = LSTM(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='lstm-layer-{}'.format(i))
        elif recurrent_type == 'GRU':
            recurrent_layer = GRU(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='gru-layer-{}'.format(i))

        if bidirectional:
            recurrent_layer = Bidirectional(
                recurrent_layer, name='bidir-' +
                recurrent_layer.name)

        model.add(recurrent_layer)

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

# Let's try creating RNN model with this helper:
batch_size = 32
embedding_dim = 20
max_seq_length = 100

train_data, valid_data, test_data, n = preprocess_datasets(
    ds_raw_train, ds_raw_valid, ds_raw_test,
    max_seq_length=max_seq_length,
    batch_size=batch_size
)

vocab_size = n + 2

rnn_model = build_rnn_model(
    embedding_dim, vocab_size,
    recurrent_type='SimpleRNN',
    n_recurrent_units=64,
    n_recurrent_layers=1,
    bidirectional=True)

rnn_model.summary()

rnn_model.compile(
    optimizer=k.optimizers.Adam(1e-3),
    loss=k.losses.BinaryCrossentropy(
        from_logits=False), metrics=['accuracy'])

history = rnn_model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10)
