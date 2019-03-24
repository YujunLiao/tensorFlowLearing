# This demo demonstrates how to read a smallest tensorflow graph.
import tensorflow as tf
# define two variables and initialize as 0
variable = tf.Variable(0)
variable2 = tf.Variable(0)

# define a constant
one = tf.constant(1, name='1')

# define add operation
add = tf.add(variable, one)

# define train operation,update variable
train = tf.assign(variable, add)


add2 = tf.add(train, one)
train2 = tf.assign(variable2, add2)

# all the Variables, have to be initialized before using
# init = tf.initialize_all_variables() # method old for initializing
# use the method behind to initialize variables
init = tf.global_variables_initializer()

# start a Session
with tf.Session() as sess:
	sess.run(init)
	for i in range(20):
		sess.run(train2)
		print(i, sess.run(variable), train, sess.run(variable2), train2)

# Write the log file for tensorboard.
# command: tensorboard --logdir \logs
# writer = tf.summary.FileWriter("./logs", sess.graph)