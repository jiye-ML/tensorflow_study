'''
tensorflow实现softmax regression 识别手写体数字
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# session
sess = tf.InteractiveSession()
# placeholder 输入数据的地方，None 表示不限条数的输入
x = tf.placeholder(tf.float32, [None, 784])

# 可以持久化的变量
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# predict
y = tf.nn.softmax(tf.matmul(x, w) + b)

# loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# optimize
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# train
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
