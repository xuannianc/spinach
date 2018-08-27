import tensorflow as tf

A = tf.placeholder(tf.float32, (None, 3), name='p')
B = A + 5

with tf.Session() as sess:
    b_val = sess.run(B, feed_dict={A: [[1, 2, 3]]})
    # feed value directly
    b_val_1 = sess.run(B, feed_dict={B: [[1, 2, 3]]})
print(b_val)
print(b_val_1)
