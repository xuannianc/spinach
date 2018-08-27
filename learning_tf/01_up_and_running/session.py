import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

# session 1

# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# print(sess.run(f))
# sess.close()

# session 2

with tf.Session() as sess:
    # set sess as the default session
    x.initializer.run()
    y.initializer.run()
    print(f.eval())

# session 3: throw exception, no default session
# tf.get_default_session().run(x.initializer)
# tf.get_default_session().run(y.initializer)
# print(tf.get_default_session.run(f))

# session 4: Interactive session
sess = tf.InteractiveSession()
# automatically sets itself as the default session
x.initializer.run()
y.initializer.run()
print(f.eval())
sess.close()