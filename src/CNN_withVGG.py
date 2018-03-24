# coding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
import scipy.io
import os
import scipy.misc


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def net(data_path, input_image):
    # 网络骨架在VGG里已经定义好，无需再定义
    # 各层，要和加载的VGG相匹配
    layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
            'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
            'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
            'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
            )
    # 加载VGG文件
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    # 得到三个通道的均值，因为该VGG制作人做了减均值操作
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    net = {}  # 字典结构，保存各层输出的结果值
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]  # 取name的前四位字符
        if kind == 'conv':  # 若为卷积层，则卷积操作
            kernels, bias = weights[i][0][0][0][0]  # 该VGG在此维[0]存放权重,[1]存放偏置
            # matconvnet:权重weights为[width,height,in_channels,out_channels]
            # tensoeflow:权重weights为[height,width,in_channels,out_channels]
            # 两者不一，要进行转换
            kernels = np.transpose(kernels, (1, 0, 2, 3))  # 按顺序转换，即此处把0和1维转换
            bias = bias.reshape(-1)  # 该VGG的偏置参数为二维，要转换成一维
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':  # 若为relu则relu操作
            current = tf.nn.relu(current)
        elif kind == 'pool':  # 若为池化层，则池化操作
            current = _pool_layer(current)
        net[name] = current  # 添加到net里面
    assert len(net) == len(layers)
    return net, mean_pixel, layers


cwd = os.getcwd()  # 获取当前路径
VGG_PATH = cwd + "/data/imagenet-vgg-verydeep-19.mat"
IMG_PATH = cwd + "/data/cat.jpg"
input_image = imread(IMG_PATH)
shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
with tf.Session() as sess:
    image = tf.placeholder('float', shape=shape)
    nets, mean_pixel, all_layers = net(VGG_PATH, image)
    # 图象预处理，该VGG做了减均值操作
    input_image_pre = np.array([preprocess(input_image, mean_pixel)])
    layers = all_layers
    # 每层可视化
    for i, layer in enumerate(layers):
        print("[%d/%d] %s" % (i + 1, len(layers), layer))
        features = nets[layer].eval(feed_dict={image:input_image_pre})
        print("Type of 'features' is ", type(features))
        print("Shape of 'features' is %s" % (features.shape,))
        # 画出特征图
        if 1:
            plt.figure(i + 1, figsize=(10, 5))
            plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i + 1)      
            plt.title("" + layer)
            plt.colorbar()
            plt.show()

