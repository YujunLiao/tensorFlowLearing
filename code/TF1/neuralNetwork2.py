"""
This demo try to train a model to distinguish the hand write digits,
but fail.


"""
import tensorflow as tf
import operator as op
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

number = 1
image_shaped_input = tf.reshape(train_images[:number], [-1, 784])
image_shaped_input = tf.cast(image_shaped_input, tf.float32)

image_labels_output = list()
for label in train_labels[:number]:
	classification = list([0]*10)
	classification[label] = 1
	image_labels_output.append(classification)
image_labels_output = tf.cast(image_labels_output, tf.float32)


def add_layer(inputs, in_size, out_size, activation_function=None,):
	# add one more layer and return the output of this layer
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b,)
	return outputs


# xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
# ys = tf.placeholder(tf.float32, [None, 10])
prediction = add_layer(image_shaped_input, 784, 10,  activation_function=tf.nn.softmax)
# loss = tf.reduce_mean(-tf.reduce_sum((image_labels_output+1) * tf.log(prediction+1), reduction_indices=[1]))
# loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(image_labels_output-prediction), 1))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(image_labels_output * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(10):

	print(i)
	print('predic:', sess.run(prediction))
	print('actual:', sess.run(image_labels_output))
	if op.eq(sess.run(prediction), sess.run(image_labels_output)).all():
		print("right")
	else:
		print("false")
	# print('loss  :', sess.run(loss))
	print('cross_entropy  :', sess.run(cross_entropy))
	sess.run(train_step)

sess.close()
