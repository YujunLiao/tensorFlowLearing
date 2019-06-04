# 深入mnist数据集

import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import warnings
import math
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def addConnect(input, input_size, output_size, activate_function=None):
    """
    Args：
        input：输入的数据
        input_size：输入的数据维度，也就是说一行几列
        output_size: 输出维度，经过这层网络要输出一行几列的数据
        activate_function： 激活函数
    desc：
        构建一层NN
    """
    weight = tf.Variable(tf.truncated_normal(
        [input_size, output_size], stddev=0.1))
    bias = tf.Variable(tf.zeros([1, output_size]))
    y = tf.matmul(input, weight) + bias
    if activate_function is None:
        return y
    else:
        return activate_function(y)

# truncated_normal 去掉了 random_normal 两边的尾巴


def test_normal_and_truncated_normal():
    """
    desc：
        测试一下tf.random_normal 和 tf.truncated_normal，这两个函数初始化weight的时候对结果有很大影响
    """
    n = 50000
    A = tf.truncated_normal((n,))
    B = tf.random_normal((n,))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a, b = sess.run([A, B])

    plt.subplot(211)
    plt.hist(a, 100, (-4.2, 4.2))
    plt.title("truncated_normal")
    plt.subplot(212)
    plt.hist(b, 100, (-4.2, 4.2))
    plt.title("random_normal")
    plt.show()


def test_mnist():
    """
    desc:
        熟悉一下mnist数据集和常用属性和方法
    """
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    # dir函数前面都是一些通用方法和继承的内置方法，我们主要用的方法和属性都在后边，所以打印了后10个
    # ['_asdict', '_fields', '_make', '_replace', '_source', 'count', 'index', 'test', 'train', 'validation']
    print("dir(mnist)：", dir(mnist)[-10:])
    """
    test:测试集
    validation:验证集
    train:训练集
    """

    # ['epochs_completed', 'images', 'labels', 'next_batch', 'num_examples']
    """
    images:图片
    labels：标签
    num_examples：图片数量
    next_batch：获取一个batch的image和label
    """
    print("dir(mnist.test)：", dir(mnist.test)[-5:])
    # 10000
    print("mnist.test.num_examples:", mnist.test.num_examples)

    test_images = mnist.test.images
    # (10000, 784) 10000张图片，28*28的，只是把二维数组变成一维数组了
    print(test_images.shape, type(test_images))

    # 用numpy的方法把784的数组转换成28 * 28的，-1表示自动，当然你也可以reshape(28,28)
    image = test_images[0].reshape(28, -1)
    plt.subplot(131)
    plt.imshow(image)
    plt.axis('off')  # 不显示坐标尺寸
    plt.subplot(132)
    plt.imshow(image, cmap='gray')  # 0-255 级灰度，0：黑色，1：白色，黑底白字；
    plt.axis('off')
    plt.subplot(133)
    # 翻转 gray 的显示，如果 gray 将图像显示为黑底白字，gray_r 会将其显示为白底黑字
    plt.imshow(image, cmap='gray_r')
    plt.axis('off')
    plt.show()


def drawDigit(position, image, title, isTrue=True):
    """
    desc:
        封装plt对image的画图

    args：
        position：展示在plt.subplot的哪个位置
        image：1 * 784 的ndarray
        title：plot的title,
        isTrue: 预测的是否是真
    """
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap="gray_r")
    plt.axis("off")
    if not isTrue:
        plt.title(title, color="red")
    else:
        plt.title(title)


def batchDraw(batch_size):
    """
    desc:
        批量图展示到plt上
    args：
        batch_size: 一次性展示多少张图，要完全平方数
    """
    # mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    # 从train获取batch_size 这么多个数据
    images, labels = mnist.train.next_batch(batch_size)

    # 有多少个图片，其实就是batch_size
    image_number = images.shape[0]

    # 想吧batch_size个图片放到一个大正方形展示，那么有sqrt(batch_size)行和列
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number

    # 指定画布尺寸
    plt.figure(figsize=(row_number, column_number))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index+1)
                image = images[index]
                title = '%d' % (np.argmax(labels[index]))
                drawDigit(position, image, title)


def predictByDNN(epochs=1000, batch_size=100):
    """
    desc:
        用NN和softmax识别mnist数据集
    args：
        epochs：训练次数
        batch_size: 一批次训练多少张图片
    """

    x = tf.placeholder(tf.float32, [None, 28*28])
    y = tf.placeholder(tf.float32, [None, 10])

    connect1 = addConnect(x, 28*28, 300, tf.nn.relu)
    connect2 = addConnect(connect1, 300, 64, tf.nn.relu)
    predict_y = addConnect(connect2, 64, 10, tf.nn.softmax)

    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predict_y), 1))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 初始化变量
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            images, labels = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: images, y: labels})
            if(epoch % 50 == 0):
                correct_prediction = tf.equal(
                    tf.argmax(predict_y, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
                accuracy_value = sess.run(
                    accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

                print('step:%d accuracy:%.4f' % (epoch, accuracy_value))

        # 用测试集 可视化一下
        images, labels = mnist.test.next_batch(batch_size)
        predict_labels = sess.run(predict_y, feed_dict={x: images, y: labels})
        image_number = images.shape[0]
        row_number = math.ceil(image_number ** 0.5)
        column_number = row_number
        plt.figure(figsize=(row_number + 10, column_number + 10))
        for i in range(row_number):
            for j in range(column_number):
                index = i * column_number + j
                if index < image_number:
                    position = (row_number, column_number, index+1)
                    image = images[index]
                    actual = np.argmax(labels[index])
                    predict = np.argmax(predict_labels[index])
                    isTrue = actual == predict
                    title = 'act:%d--pred:%d' % (actual, predict)
                    drawDigit(position, image, title, isTrue)


def test_argmax_equal_cast():
    """
    desc:
        测试tf.argmax, tf.equal, tf.cast函数
    """
    x1 = np.array([[1, 2, 3], [1, 5, 6], [21.3, 6, 7]])
    x2 = np.array([[1, 4, 3], [1, 5, 6], [21.3, 6, 7]])
    # argmax 是获取最大值的下标，axis=1是按行获取，axis=0是按列获取
    y1_argmax = tf.argmax(x1, 1)
    y2_argmax = tf.argmax(x2, 1)
    y_equal = tf.equal(y1_argmax, y2_argmax)
    y_cast = tf.cast(y_equal, tf.float32)
    y_reduce_mean = tf.reduce_mean(y_cast)
    with tf.Session() as sess:
        # [2 2 0]
        print(sess.run(y1_argmax))
        # [1 2 0]
        print(sess.run(y2_argmax))
        # [False  True  True]
        print(sess.run(y_equal))
        # [0. 1. 1.] 把True变成1.0 False变成0.0
        print(sess.run(y_cast))
        # 0.6666667 取平均值，即True所占的比例
        print(sess.run(y_reduce_mean))


if __name__ == "__main__":
    predictByDNN()
    plt.show()