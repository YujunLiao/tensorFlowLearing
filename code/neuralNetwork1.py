# The first neural network demo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Define the function of add a new layer
def add_layer(inputs, in_size, out_size, layerName, activation_function=None):
    # tf.name_scope usage
    with tf.name_scope(layerName):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            # result visualization, record the change of the variable
            tf.summary.histogram('weight1', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            # result visualization, record the change of the variable
            tf.summary.histogram('biae1',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        # result visualization, record the change of the variable
        # tf.summary.histogram(layerName + '/outputs', outputs)
        tf.summary.histogram(layerName+'/output', outputs)
    return outputs


# Create training data.
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# Input scope
with tf.name_scope('inputsTrainingData'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hide layer and output layer
# one input, one output
# ten neural on hide layer
hideLayer = add_layer(xs, 1, 10, 'hide', activation_function=tf.nn.relu)
predictionLayer = add_layer(hideLayer, 10, 1, 'prediction', activation_function=None)
# compute loss
with tf.name_scope('lossCompute'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predictionLayer), reduction_indices=[1]))
    # result visualization, record the change of the variable
    tf.summary.scalar('loss', loss)
# define the train step to minimize the loss.
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialize all variables
init_op = tf.global_variables_initializer()
# usage odd
# init_op = tf.initialize_all_variables()


with tf.Session() as sess:
    # result visualization
    # merge all the summaries
    merged = tf.summary.merge_all()

    # write the log file for the tensorflow
    # used on pi
    # writer = tf.train.SummaryWriter("./logs", sess.graph)
    # used on computer
    writer = tf.summary.FileWriter("./logs", sess.graph)

    sess.run(init_op)

    # create a figure
    fig = plt.figure()
    # set title
    fig.suptitle('test title')

    # divide the figure into 3*2=6 parts, there are 3 rows, 2 columns
    # this subplot will be drawn on the 5th part.
    # ax = fig.add_subplot(3, 2, 5)
    subFig = fig.add_subplot(1, 1, 1)
    # draw
    subFig.scatter(x_data, y_data)
    # show the figure
    # block=False indicates that the figure can change
    # plt.show(block=False)

    # train for 1000 times
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # result visualization
            rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(rs, i)
            try:
                print(i)
                # remove a line according to the name
                # for example,subFig.lines.remove(Line2D(_line0))
                # subFig.lines.remove(line[0])
                subFig.lines.remove(subFig.lines[0])
            except Exception:
                pass
            prediction_value = sess.run(predictionLayer, feed_dict={xs: x_data})

            # Add a line into the axes.
            # Return a list that has only the current Line2D type object.
            # line = subFig.plot(x_data, prediction_value, 'r-', lw=1)
            subFig.plot(x_data, prediction_value, 'r-', lw=1)
            plt.show(block=False)
            plt.pause(0.5)

            # example
            # print(line)
            # [<matplotlib.lines.Line2D object at 0x000001F30D245978>]
            # print(line[0])
            # Line2D(_line2)

            # Return a list that has all the Line2D type objects in this subFig.
            # subFig.lines

            # example
            # print(subFig.lines)
            # [<matplotlib.lines.Line2D object at 0x000001F30D0E09B0>,
            #  <matplotlib.lines.Line2D object at 0x000001F30B0555C0>,
            # <matplotlib.lines.Line2D object at 0x000001F30D245978>]
            # print(subFig.lines[0])
            # Line2D(_line0)
            # print(subFig.lines[2])
            # Line2D(_line2)

            # print(subFig.lines, subFig.lines[0], len(subFig.lines))
            # print(line, line[0], len(line))

    writer.close()




