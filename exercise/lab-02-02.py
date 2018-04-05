import tensorflow as tf
tf.set_random_seed(777)

# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Now we can use X and Y in place of x_data and y_data
# 'placeholder' for a tensor that will be always fed using feed_dict
# See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis XW + b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    # W_val, b_val = sess.run([W, b])
    #
    # _X = [1, 2, 3, 4, 5]
    # _Y = [2.1, 3.1, 4.1, 5.1, 6.1]
    #
    # cost_val = sess.run(cost, feed_dict={X: _X, Y: _Y})
    # sess.run(train, feed_dict={X: _X, Y: _Y})

    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X: [1, 2, 3, 4, 5],
                                                    Y: [2.1, 3.1, 4.1, 5.1, 6.1]})


    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)


# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))


