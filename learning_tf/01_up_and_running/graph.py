import tensorflow as tf

# All nodes we created are automatically added to the default graph
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

with tf.Graph().as_default():
    x2 = tf.Variable(2)
print(x2.graph is tf.get_default_graph())
