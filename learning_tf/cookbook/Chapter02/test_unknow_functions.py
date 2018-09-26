import tensorflow as tf

"""
gather_nd: indexing 一个 tensor
"""
x = tf.random_normal([13, 100])
# 取 x 第 5,10,12 个 元素组成一个新的 tensor, y
y = tf.gather_nd(x,[[5], [10], [12]])
# stack 在此处并不起任何作用
z = tf.stack(y)

with tf.Session() as sess:
    print(sess.run([y,z]))
