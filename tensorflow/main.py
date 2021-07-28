import tensorflow as tf
import numpy as np

np.set_printoptions(precision=3)

a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]

t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a)
print(t_b)

t_ones = tf.ones((2, 3))
print(t_ones.shape)
print(t_ones.numpy())

const_tensor = tf.constant([1.2, 5, np.pi], dtype=tf.float32)
print(const_tensor)

# casting types
t_a_new = tf.cast(t_a, tf.int64)
print(t_a_new.dtype)

# transposing a tensor
t = tf.random.uniform(shape=(3, 5))
t_tr = tf.transpose(t)
print(t.shape, ' ---> ', t_tr.shape)

# Reshaping a tensor
t = tf.zeros((30,))
t_reshape = tf.reshape(t, shape=(5, 6))
print(t_reshape.shape)

# Removing dimensions
t = tf.zeros((1, 2, 1, 4, 1))
t_sqz = tf.squeeze(t, axis=(2, 4))
print(t.shape, ' --> ', t_sqz.shape)

# Mathematical operations on tensors
tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5,2), minval=-1.0, maxval=1.0)
t2 = tf.random.normal(shape=(5,2), mean=0.0, stddev=1.0)

# element-wise product of two size-compatible tensors
t3 = tf.multiply(t1, t2).numpy()
print(t3)

# reduce to aggregate value along certain axis
# mean of columns:
t4 = tf.math.reduce_mean(t1, axis=0)
print(t4)

# matrix-matrix product with transposing second matrix
t5 = tf.linalg.matmul(t1, t2, transpose_b=True)
print(t5.numpy())

# matrix-matrix product with transposing first matrix
t6 = tf.linalg.matmul(t1, t2, transpose_a=True)
print(t6.numpy())

# calculating Lp norm of a tensor
# L2:
norm_t1 = tf.norm(t1, ord=2, axis=1).numpy()
print(norm_t1)

# Split, stack and concatenate tensors
tf.random.set_seed(1)
t = tf.random.uniform((6,))
print(t.numpy())

t_splits = tf.split(t, num_or_size_splits=3)
[item.numpy() for item in t_splits]

tf.random.set_seed(1)
t = tf.random.uniform((5,))
print(t.numpy())

t_splits = tf.split(t, num_or_size_splits=[3, 2])
[item.numpy() for item in t_splits]

# stacking:
A = tf.ones((3,))
B = tf.zeros((2,))
C = tf.concat([A, B], axis=0)
print(C.numpy())

A = tf.ones((3,))
B = tf.zeros((3,))
C = tf.stack([A, B], axis=1)
print(C.numpy())
