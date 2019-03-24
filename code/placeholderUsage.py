# This file demonstrate the usage of placeholder function.
import tensorflow as tf

# define the type of placeholder
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# multiply the two variables
ouput = tf.multiply(input1, input2)

x1=input('x1:')
x2=input('x2:')
with tf.Session() as sess:
	print(sess.run(ouput, feed_dict={input1: x1, input2: x2}))
# print(sess.run(ouput, feed_dict={input1: [x1], input2: [x2]}))
