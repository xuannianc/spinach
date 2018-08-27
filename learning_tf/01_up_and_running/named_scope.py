import tensorflow as tf

with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
print(error.op.name)
print(mse.op.name)

a1 = tf.Variable(0, name="a")  # name == "a"
a2 = tf.Variable(0, name="a")  # name == "a_1"

with tf.name_scope("param"):  # name == "param"
    a3 = tf.Variable(0, name="a")  # name == "param/a"

with tf.name_scope("param"):  # name == "param_1"
    a4 = tf.Variable(0, name="a")  # name == "param_1/a"

for node in (a1, a2, a3, a4):
    print(node.op.name)
