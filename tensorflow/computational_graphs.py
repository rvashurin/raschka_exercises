import tensorflow as tf

## TF v1.x style
g = tf.Graph()

with g.as_default():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.constant(3, name='c')
    z = 2*(a - b) + c

with tf.compat.v1.Session(graph=g) as sess:
    print('Result: z=', sess.run(z))

## TF v2 style
a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = tf.constant(3, name='c')
z = 2*(a - b) + c
tf.print('Result: z=', z)

## Loading input data TF v1.x style
g = tf.Graph()
with g.as_default():
    a = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_a')
    b = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_b')
    c = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_c')
    z = 2*(a - b) + c

with tf.compat.v1.Session(graph=g) as sess:
    feed_dict = {a:1, b:2, c:3}
    print('Result: z=', sess.run(z, feed_dict=feed_dict))

## Loading input data TF v2.x style
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

tf.print('Scalar Inputs:', compute_z(1, 2, 3))
tf.print('Rank 1 Inputs:', compute_z([1], [2], [3]))
tf.print('Rank 2 Inputs:', compute_z([[1]], [[2]], [[3]]))

## Compiling dynamic graphs to static ones
@tf.function
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

tf.print('Scalar Inputs:', compute_z(1, 2, 3))
tf.print('Rank 1 Inputs:', compute_z([1], [2], [3]))
tf.print('Rank 2 Inputs:', compute_z([[1]], [[2]], [[3]]))

# Limit input types for graph
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32)))
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z = tf.add(r2, c)
    return z

tf.print('Rank 1 Inputs:', compute_z([1], [2], [3]))
tf.print('Rank 1 Inputs:', compute_z([1, 2], [2, 4], [3, 6]))

# These will cause errors
# tf.print('Rank 0 Inputs:', compute_z(1, 2, 3))
# tf.print('Rank 2 Inputs:', compute_z([[1], [2]],
#                                      [[2], [4]],
#                                      [[3], [6]]))

## TF variables:
a = tf.Variable(initial_value=3.14, name='var_a')
print(a)

b = tf.Variable(initial_value=[1, 2, 3], name='var_a')
print(b)

c = tf.Variable(initial_value=[True, False], name='var_a')
print(c)

d = tf.Variable(initial_value=['abc'], name='var_a')
print(d)

# non-trainable vars:
w = tf.Variable([1, 2, 3], trainable=False)
print(w.trainable)

# assigning vars:
print(w.assign([3, 1, 4], read_value=True))
w.assign_add([2, -1, 2], read_value=False)
print(w.value())

# random initialization
tf.random.set_seed(1)
init = tf.keras.initializers.GlorotNormal()
tf.print(init(shape=(3,)))

v = tf.Variable(init(shape=(2, 3)))
tf.print(v)

# within Module:
class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2, 3)))
        self.w2 = tf.Variable(init(shape=(1, 2)), trainable=False)

m = MyModule()
print('All module variables:', [v.shape for v in m.variables])
print('Trainable variables:', [v.shape for v in m.trainable_variables])

# Can't define variables inside decorated functions:
# @tf.function
# def f(x):
#     w = tf.Variable([1, 2, 3])
#
# f([1])

w = tf.Variable(tf.random.uniform((3, 3)))
@tf.function
def compute_z(x):
    return tf.matmul(w, x)

x = tf.constant([[1], [2], [3]], dtype=tf.float32)
tf.print(compute_z(x))
