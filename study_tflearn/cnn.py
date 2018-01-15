# encoding: UTF-8
import os
import time
import shutil
import zipfile
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
import scipy.misc as misc
import tensorflow.contrib.layers as tcl
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    pass


class PreData:

    def __init__(self, zip_file, ratio=20):
        data_path = zip_file.split(".zip")[0]
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")

        if not os.path.exists(data_path):
            f = zipfile.ZipFile(zip_file, "r")
            f.extractall(data_path)

            all_image = self.get_all_images(os.path.join(data_path, data_path.split("/")[-1]))
            self.get_data_result(all_image, ratio, Tools.new_dir(self.train_path), Tools.new_dir(self.test_path))
        else:
            Tools.print_info("data is exists")
        pass

    # 生成测试集和训练集
    @staticmethod
    def get_data_result(all_image, ratio, train_path, test_path):
        train_list = []
        test_list = []

        # 遍历
        Tools.print_info("bian")
        for now_type in range(len(all_image)):
            now_images = all_image[now_type]
            for now_image in now_images:
                # 划分
                if np.random.randint(0, ratio) == 0:  # 测试数据
                    test_list.append((now_type, now_image))
                else:
                    train_list.append((now_type, now_image))
            pass

        # 打乱
        Tools.print_info("shuffle")
        np.random.shuffle(train_list)
        np.random.shuffle(test_list)

        # 提取训练图片和标签
        Tools.print_info("train")
        for index in range(len(train_list)):
            now_type, image = train_list[index]
            shutil.copyfile(image, os.path.join(train_path,
                                                str(np.random.randint(0, 1000000)) + "-" + str(now_type) + ".jpg"))

        # 提取测试图片和标签
        Tools.print_info("test")
        for index in range(len(test_list)):
            now_type, image = test_list[index]
            shutil.copyfile(image, os.path.join(test_path,
                                                str(np.random.randint(0, 1000000)) + "-" + str(now_type) + ".jpg"))

        pass

    # 所有的图片
    @staticmethod
    def get_all_images(images_path):
        all_image = []
        all_path = os.listdir(images_path)
        for one_type_path in all_path:
            now_path = os.path.join(images_path, one_type_path)
            if os.path.isdir(now_path):
                now_images = glob(os.path.join(now_path, '*.jpg'))
                all_image.append(now_images)
            pass
        return all_image

    # 生成数据
    @staticmethod
    def main(zip_file):
        pre_data = PreData(zip_file)
        return pre_data.train_path, pre_data.test_path

    pass


class Data:
    def __init__(self, type_number, image_size, image_channel, train_path, can_load_image_percent):

        self.type_number = type_number
        self.image_size = image_size
        self.image_channel = image_channel

        self._train_images = glob(os.path.join(train_path, "*.jpg"))
        # 加载图片的比例 ， 全部加载会内存不够用
        self._can_load_image_percent = can_load_image_percent
        self.batch_size = int(len(self._train_images) * self._can_load_image_percent)

        pass

    def next_train(self):
        begin = np.random.randint(0, len(self._train_images) - self.batch_size)
        return self.norm_image_label(self._train_images[begin: begin + self.batch_size])

    def norm_image_label(self, images_list):
        images = [np.array(misc.imread(image_path).astype(np.float)) / 255.0 for image_path in images_list]
        labels = [int(image_path.split("-")[1].split(".")[0]) for image_path in images_list]
        return images,self.one_hot(labels)

    def one_hot(self, label):
        labels = np.zeros(shape=[len(label), self.type_number], dtype=np.uint32)
        for index, value in enumerate(label):
            labels[index][value] = 1
        return labels

    pass


class AlexNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    def alex_net(self, input_op):
        #  256 X 256 X 3
        with tf.name_scope("conv1") as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, self._image_channel, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(input=input_op, filter=kernel, strides=[1, 4, 4, 1], padding="SAME")  # 64 X 64 X 64
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32))
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn1")
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")  # 31 X 31 X 64

        with tf.name_scope("conv2") as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")  # 31 X 31 X 192
            biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32))
            conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn2")
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")  # 15 X 15 X 192

        with tf.name_scope("conv3") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")  # 15 X 15 X 384
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
            conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope("conv4") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")  # 15 X 15 X 256
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32))
            conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        with tf.name_scope("conv5") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")  # 15 X 15 X 256
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32))
            conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")  # 7 X 7 X 256

        dim = pool3.get_shape()[1].value * pool3.get_shape()[2].value * pool3.get_shape()[3].value
        reshape = tf.reshape(pool3, [-1, dim])

        with tf.name_scope("fc1") as scope:
            weights = tf.Variable(tf.truncated_normal([dim, 384], dtype=tf.float32, stddev=1e-1))
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
            fc1 = tf.nn.relu(tf.add(tf.matmul(reshape, weights), biases), name=scope)  # dim X 384

        with tf.name_scope("fc2") as scope:
            weights = tf.Variable(tf.truncated_normal([384, 192], dtype=tf.float32, stddev=1e-1))
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
            fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weights), biases), name=scope)  # 384 X 192

        with tf.name_scope("fc3") as scope:
            weights = tf.Variable(tf.truncated_normal([192, self._type_number], dtype=tf.float32, stddev=1e-1))
            biases = tf.Variable(tf.constant(0.0, shape=[self._type_number], dtype=tf.float32))
            logits = tf.add(tf.matmul(fc2, weights), biases, name=scope)  # 192 X number_type

        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)
        return logits, softmax, prediction

    pass


class VGGNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    # 网络
    # keep_prob=0.7
    def vgg_16(self, input_op, **kw):
        first_out = 32

        conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        conv_1_2 = self._conv_op(conv_1_1, "conv_1_2", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        pool_1 = self._max_pool_op(conv_1_2, "pool_1", 2, 2, stripe_height=2, stripe_width=2)

        conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        conv_2_2 = self._conv_op(conv_2_1, "conv_2_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        pool_2 = self._max_pool_op(conv_2_2, "pool_2", 2, 2, stripe_height=2, stripe_width=2)

        conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_2 = self._conv_op(conv_3_1, "conv_3_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_3 = self._conv_op(conv_3_2, "conv_3_3", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_3 = self._max_pool_op(conv_3_3, "pool_3", 2, 2, stripe_height=2, stripe_width=2)

        conv_4_1 = self._conv_op(pool_3, "conv_4_1", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_2 = self._conv_op(conv_4_1, "conv_4_2", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_3 = self._conv_op(conv_4_2, "conv_4_3", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        pool_4 = self._max_pool_op(conv_4_3, "pool_4", 2, 2, stripe_height=2, stripe_width=2)

        conv_5_1 = self._conv_op(pool_4, "conv_5_1", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_5_2 = self._conv_op(conv_5_1, "conv_5_2", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_5_3 = self._conv_op(conv_5_2, "conv_5_3", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        pool_5 = self._max_pool_op(conv_5_3, "pool_5", 2, 2, stripe_height=2, stripe_width=2)

        shp = pool_5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        reshape_pool_5 = tf.reshape(pool_5, [-1, flattened_shape], name="reshape_pool_5")

        fc_6 = self._fc_op(reshape_pool_5, name="fc_6", n_out=2048)
        fc_6_drop = tf.nn.dropout(fc_6, keep_prob=kw["keep_prob"], name="fc_6_drop")

        fc_7 = self._fc_op(fc_6_drop, name="fc_7", n_out=1024)
        fc_7_drop = tf.nn.dropout(fc_7, keep_prob=kw["keep_prob"], name="fc_7_drop")

        fc_8 = self._fc_op(fc_7_drop, name="fc_8", n_out=self._type_number)
        softmax = tf.nn.softmax(fc_8)
        prediction = tf.argmax(softmax, 1)

        return fc_8, softmax, prediction

    # 网络
    # keep_prob=0.7
    def vgg_12(self, input_op, **kw):
        first_out = 32

        conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        conv_1_2 = self._conv_op(conv_1_1, "conv_1_2", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        pool_1 = self._max_pool_op(conv_1_2, "pool_1", 2, 2, stripe_height=2, stripe_width=2)

        conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        conv_2_2 = self._conv_op(conv_2_1, "conv_2_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        pool_2 = self._max_pool_op(conv_2_2, "pool_2", 2, 2, stripe_height=2, stripe_width=2)

        conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_2 = self._conv_op(conv_3_1, "conv_3_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_3 = self._max_pool_op(conv_3_2, "pool_3", 2, 2, stripe_height=2, stripe_width=2)

        conv_4_1 = self._conv_op(pool_3, "conv_4_1", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_2 = self._conv_op(conv_4_1, "conv_4_2", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_3 = self._conv_op(conv_4_2, "conv_4_3", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        pool_4 = self._max_pool_op(conv_4_3, "pool_4", 2, 2, stripe_height=2, stripe_width=2)

        shp = pool_4.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        reshape_pool_4 = tf.reshape(pool_4, [-1, flattened_shape], name="reshape_pool_4")

        fc_5 = self._fc_op(reshape_pool_4, name="fc_5", n_out=2048)
        fc_5_drop = tf.nn.dropout(fc_5, keep_prob=kw["keep_prob"], name="fc_5_drop")

        fc_6 = self._fc_op(fc_5_drop, name="fc_6", n_out=1024)
        fc_6_drop = tf.nn.dropout(fc_6, keep_prob=kw["keep_prob"], name="fc_6_drop")

        fc_7 = self._fc_op(fc_6_drop, name="fc_7", n_out=self._type_number)
        softmax = tf.nn.softmax(fc_7)
        prediction = tf.argmax(softmax, 1)

        return fc_7, softmax, prediction

    # 网络
    # keep_prob=0.7
    def vgg_10(self, input_op, **kw):
        first_out = 32

        conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        conv_1_2 = self._conv_op(conv_1_1, "conv_1_2", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        pool_1 = self._max_pool_op(conv_1_2, "pool_1", 2, 2, stripe_height=2, stripe_width=2)

        conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        conv_2_2 = self._conv_op(conv_2_1, "conv_2_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        pool_2 = self._max_pool_op(conv_2_2, "pool_2", 2, 2, stripe_height=2, stripe_width=2)

        conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_2 = self._conv_op(conv_3_1, "conv_3_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_3 = self._conv_op(conv_3_2, "conv_3_3", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_3 = self._max_pool_op(conv_3_3, "pool_3", 2, 2, stripe_height=2, stripe_width=2)

        reshape_pool_3 = tf.reshape(pool_3, [self._batch_size, -1], name="reshape_pool_3")

        fc_4 = self._fc_op(reshape_pool_3, name="fc_4", n_out=2048)
        fc_4_drop = tf.nn.dropout(fc_4, keep_prob=kw["keep_prob"], name="fc_4_drop")

        fc_5 = self._fc_op(fc_4_drop, name="fc_5", n_out=1024)
        fc_5_drop = tf.nn.dropout(fc_5, keep_prob=kw["keep_prob"], name="fc_5_drop")

        fc_6 = self._fc_op(fc_5_drop, name="fc_6", n_out=self._type_number)
        softmax = tf.nn.softmax(fc_6)
        prediction = tf.argmax(softmax, 1)

        return fc_6, softmax, prediction

    # 创建卷积层
    @staticmethod
    def _conv_op(input_op, name, kernel_height, kernel_width, n_out, stripe_height, stripe_width):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name=name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[kernel_height, kernel_width, n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, filter=kernel, strides=(1, stripe_height, stripe_width, 1), padding="SAME")
            biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
            activation = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
            return activation
        pass

    # 创建全连接层
    @staticmethod
    def _fc_op(input_op, name, n_out):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())
            biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name="b")
            activation = tf.nn.relu_layer(x=input_op, weights=kernel, biases=biases, name=scope)
            return activation
        pass

    # 最大池化层
    @staticmethod
    def _max_pool_op(input_op, name, kernel_height, kernel_width, stripe_height, stripe_width):
        return tf.nn.max_pool(input_op, ksize=[1, kernel_height, kernel_width, 1],
                              strides=[1, stripe_height, stripe_width, 1], padding="SAME", name=name)

    pass


class CNNNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass
    
    # 网络
    def cnn_5(self, input_op):
        weight_1 = tf.Variable(tf.truncated_normal(shape=[5, 5, self._image_channel, 64], stddev=5e-2))
        kernel_1 = tf.nn.conv2d(input_op, weight_1, [1, 1, 1, 1], padding="SAME")
        bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
        conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding="SAME")
        norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        weight_2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 128], stddev=5e-2))
        kernel_2 = tf.nn.conv2d(norm_1, weight_2, [1, 1, 1, 1], padding="SAME")
        bias_2 = tf.Variable(tf.constant(0.1, shape=[128]))
        conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
        norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding="SAME")

        weight_23 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))
        kernel_23 = tf.nn.conv2d(pool_2, weight_23, [1, 2, 2, 1], padding="SAME")
        bias_23 = tf.Variable(tf.constant(0.1, shape=[256]))
        conv_23 = tf.nn.relu(tf.nn.bias_add(kernel_23, bias_23))
        norm_23 = tf.nn.lrn(conv_23, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_23 = tf.nn.max_pool(norm_23, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="SAME")

        reshape = tf.reshape(pool_23, [self._batch_size, -1])
        dim = reshape.get_shape()[1].value

        weight_4 = tf.Variable(tf.truncated_normal(shape=[dim, 192 * 2], stddev=0.04))
        bias_4 = tf.Variable(tf.constant(0.1, shape=[192 * 2]))
        local_4 = tf.nn.relu(tf.matmul(reshape, weight_4) + bias_4)

        weight_5 = tf.Variable(tf.truncated_normal(shape=[192 * 2, self._type_number], stddev=1 / 192.0))
        bias_5 = tf.Variable(tf.constant(0.0, shape=[self._type_number]))
        logits = tf.add(tf.matmul(local_4, weight_5), bias_5)

        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)

        return logits, softmax, prediction

    pass


class GoogleNet:
    def __init__(self, class_number, image_size, image_channel):
        # input
        self._class_number = class_number
        self._image_size = image_size
        self._image_channel = image_channel
        pass

    # 建立网络
    def network(self):
        network = input_data(shape=[None, self._image_size, self._image_size, self._image_channel])
        con1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
        pool1_3_3 = max_pool_2d(con1_7_7, 3, strides=2)
        pool1_3_3 = batch_normalization(pool1_3_3)
        conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
        conv2_3_3 = batch_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

        # 3a
        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu',
                                   name='inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_3a_5_5_reduce')
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu',
                                   name='inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool_1_1')
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool],
                                    mode='concat', axis=3)

        # 3b
        inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu',
                                          name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu',
                                   name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu',
                                          name='inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, name='inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_3b_pool_1_1')
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                    mode='concat',
                                    axis=3, name='inception_3b_output')
        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

        # 4a
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu',
                                          name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu',
                                   name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu',
                                   name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1, name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4a_pool_1_1')
        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                    mode='concat', axis=3, name='inception_4a_output')

        # 4b
        inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4b_1_1')
        inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu',
                                          name='inception_4b_3_3')
        inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu',
                                   name='inception_4b_3_3')
        inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu',
                                          name='inception_4b_5_5_reduce')
        inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4b_5_5')
        inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1, name='inception_4b_pool')
        inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4b_pool_1_1')
        inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                    mode='concat', axis=3, name='inception_4b_output')

        # 4c
        inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
        inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',
                                          name='inception_4c_3_e_reduce')
        inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu',
                                   name='inception_4c_3_3')
        inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu',
                                          name='inception_4c_5_5_reduce')
        inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4c_5_5')
        inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
        inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4c_pool_1_1')
        inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                    mode='concat', axis=3, name='inception_4c_output')

        # 4d
        inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
        inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu',
                                          name='inception_4d_3_3_reduce')
        inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu',
                                   name='inception_4d_3_3')
        inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu',
                                          name='inception_4d_5_5_reduce')
        inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4d_5_5')
        inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1, name='inception_4d_pool')
        inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4d_pool_1_1')
        inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                    mode='concat', axis=3, name='inception_4d_output')

        # 4e
        inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
        inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu',
                                          name='inception_4e_3_3_reduce')
        inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_4e_3_3')
        inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu',
                                          name='inceptin_4e_5_5_reduce')
        inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_4e_5_5')
        inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1, name='inception_4e_pool')
        inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu',
                                        name='inception_4e_pool_1_1')
        inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1],
                                    mode='concat', axis=3, name='inception_4e_output')
        pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool4_3_3')

        # 5a
        inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
        inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu',
                                          name='inception_5a_3_3_reduce')
        inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_5a_3_3')
        inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu',
                                          name='inception_5a_5_5_reduce')
        inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5a_5_5')
        inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1, name='inception_5a_pool')
        inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5a_pool_1_1')
        inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],
                                    mode='concat', axis=3, name='inception_5a_output')

        # 5b
        inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
        inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu',
                                          name='inception_5b_3_3_reduce')
        inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu',
                                   name='inception_5b_3_3')
        inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu',
                                          name='inception_5b_5_5_reduce')
        inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5b_5_5')
        inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1, name='inception_5b_pool')
        inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5b_pool_1_1')
        inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],
                                    mode='concat', axis=3, name='inception_5b_output')
        pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
        pool5_7_7 = dropout(pool5_7_7, 0.4)

        softmax = fully_connected(pool5_7_7, self._class_number, activation='softmax')
        network = regression(softmax, optimizer='adam', loss='categorical_crossentropy',learning_rate=0.001)
        return network
        pass

    pass


class Runner:
    def __init__(self, data, classifier, model_dir, model_name, batch_size):
        # data
        self._data = data
        self._type_number = self._data.type_number
        self._image_size = self._data.image_size
        self._image_channel = self._data.image_channel
        self._batch_size = batch_size

        # model
        self._classifier = classifier
        self._model_dir = model_dir
        self._model_name = model_name
        self._is_training = self.is_model_exist()
        self._model = tflearn.DNN(self._classifier.network())

        pass

    def run(self, **kwargs):
        if self._is_training:
            self.train(**kwargs)
            self.save_model()
        self.prediction()
        pass

    def train(self, **kwargs):
        # train
        image, label = self._data.next_train()
        self._model.fit(X_inputs=image, Y_targets=label, n_epoch=kwargs["n_epoch"], validation_set=0.2,
                        batch_size=self._batch_size, snapshot_step=kwargs["snapshot_step"],
                        show_metric=True, shuffle=True,)
        pass

    def save_model(self):
        self._model.save("{}/{}.tfl".format(self._model_dir, self._model_name))
        pass

    # 模型是否已经存在
    def is_model_exist(self):
        flag = not os.path.exists("{}/{}.tfl.index".format(self._model_dir, self._model_name))
        return flag
        pass

    # TODO
    def prediction(self):
        if not self._is_training:
            self._model.load("{}/{}.tfl".format(self._model_dir, self._model_name))
        # data

        pass

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="vgg", help="name")
    parser.add_argument("-epochs", type=int, default=20, help="train epoch number")
    parser.add_argument("-batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-type_number", type=int, default=45, help="type number")
    parser.add_argument("-image_size", type=int, default=256, help="image size")
    parser.add_argument("-image_channel", type=int, default=3, help="image channel")
    parser.add_argument("-can_load_image_percent", type=float, default=0.1, help="per step image percent")
    parser.add_argument("-zip_file", type=str, default="./data/resisc45.zip", help="zip file path")
    args = parser.parse_args()

    output = "name={},epochs={},batch_size={},type_number={},image_size={},image_channel={},zip_file={}" \
             "can load image percent={}"
    Tools.print_info(output.format(args.name, args.epochs, args.batch_size, args.type_number,
                                   args.image_size, args.image_channel, args.zip_file, args.can_load_image_percent))


    model_dir = "./dist/model"
    model_name = "train"

    now_train_path, now_test_path = PreData.main(zip_file=args.zip_file)
    now_data = Data(type_number=args.type_number, image_size=args.image_size, image_channel=args.image_channel,
                    train_path=now_train_path, can_load_image_percent=args.can_load_image_percent)

    googleNet = GoogleNet(args.type_number, args.image_size, args.image_channel)

    runner = Runner(data=now_data, classifier=googleNet, batch_size=args.batch_size,
                    model_dir=Tools.new_dir(model_dir), model_name=Tools.new_dir(model_name))
    runner.run(snapshot_step=100, n_epoch=args.epochs)


    pass
