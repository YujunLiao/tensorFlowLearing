# Demonstrates the smallest neural network
import tensorflow as tf
import numpy as np

# Define the function of add a new layer
def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	Wx = tf.matmul(inputs, Weights, name='matmul')
	return Wx


# Create data.
x_data = np.linspace(-1, 1, 1, dtype=np.float32)[:, np.newaxis]
y_data = x_data

# Define place holder
xs = tf.placeholder(tf.float32, [None, 1], name='xs')
ys = tf.placeholder(tf.float32, [None, 1], name='ys')

# Add layer.One input, one hide layer, one output layer.
# One input neural, one hide neural, one output neural.
l1 = add_layer(xs, 1, 1, activation_function=None)
prediction = add_layer(l1, 1, 1, activation_function=None)

# define the loss
loss =tf.square(ys - prediction)
# tran step aim to minimize the loss.
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
# initialize all the variables
init = tf.global_variables_initializer()
# start a sess
with tf.Session() as sess:
	sess.run(init)
	# train 100 times
	for i in range(100):
		# training
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
		if i % 5 == 0:
			# to see the step improvement
			print('loss:', sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

# write the log files.
writer = tf.summary.FileWriter("./logs", sess.graph)