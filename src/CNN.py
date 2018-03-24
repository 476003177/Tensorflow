# coding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
import input_data

# 加载mnist数据
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print("MNIST loaded")

# 参数设置
n_input = 784  # 输入图象为784像素
n_output = 10  # 10分类任务
weights = {
    # 设定tf变量，用高斯分布初始化
    # 第一层卷积层filter卷积核为3✖3，输入为1层特征图，输出64层特征图
    'wc1':tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    # 第二层卷积层卷积核为3✖3，输入为64层特征图，输出128层特征图
    'wc2':tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    # 第一层全连接层
    # 原图为28✖28像素，卷积层有padding存在不缩小图象
    # 每层池化层为2✖2，第二层池化层输出7✖7✖128
    # 自定义输出1024向量
    'wd1':tf.Variable(tf.random_normal([7 * 7 * 128, 1024], stddev=0.1)),
    # 第二层全连接层，接收第一层输出的1024向量，输出n_output向量，以便分类
    'wd2':tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
    }
biases = {
    # 第一层卷积层，共享参数，输出64层特征图，共有64个b偏置
    'bc1':tf.Variable(tf.random_normal([64], stddev=0.1)),
    # 第二层卷积层，共享参数，输出128层特征图，共有64个b偏置
    'bc2':tf.Variable(tf.random_normal([128], stddev=0.1)),
    # 第一层全连接层，共享参数
    'bd1':tf.Variable(tf.random_normal([1024], stddev=0.1)),
    # 第二层全连接层，共享参数
    'bd2':tf.Variable(tf.random_normal([n_output], stddev=0.1)),
    }


# 定义卷积神经网络
def conv_basic(_input, _w, _b, _keepratio):
    # 输入，tf输入数据为四维格式[batch,h高,w宽,c通道数]
    # -1指的是让tf自己推断该维长度
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    
    # 第一层卷积层
    # strides为四维格式[batch,h高,w宽,c通道数]
    # padding只有两种模式：“SAME”(默认填充)和“VALID”(不填充，无法卷积的抛弃)
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # print(help(tf.nn.conv2d))  #查看帮助文档
    # 使用relu激活函数，并设置偏置
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    # 第一层池化层，池化方式选择maxpool，池化2✖2
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout操作
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    
    # 第二层卷积层
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    
    # 第一层全连接层
    # 利用_w['wd1']进行整理输入数据
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
    # 全连接操作
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    
    # 第二层全连接层
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    out = {'input_r':_input_r, 'conv1':_conv1, 'pool1':_pool1, 'pool_dr1':_pool_dr1,
           'conv2':_conv2, 'pool2':_pool2, 'pool_dr2':_pool_dr2,
           'dense1':_dense1, 'fc1':_fc1, 'fc_dr1':_fc_dr1, 'out':_out
           }
    return out


# 定义tf变量
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# 定义运行流程
# CNN模型输出值
_pred = conv_basic(x, weights, biases, keepratio)['out']
# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=y))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, _pred))
# 优化目标
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# 判断，将分类的每个维度和y进行判断对错，argmax为判断的维度
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
# 精度
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
# 初始化
init = tf.global_variables_initializer()
# 定义Saver1️以保存模型
save_step = 1
saver = tf.train.Saver(max_to_keep=3)  # max_to_keep为保留的模型数
do_train = 0
# 定义一个Session
sess = tf.Session()
# 实际运行init初始化
sess.run(init)

training_epochs = 15  # 重复次数
batch_size = 16
display_step = 1
if do_train == 1:
    for epoch in range(training_epochs):
        avg_cost = 0;
        # batch数目
        # total_batch=int(mnist.train._num_examples/batch_size)
        total_batch = 10
        for i in range(total_batch):
            # 获取该batch的数据，batch_xs为数据，batch_ys为标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行
            sess.run(optm, feed_dict={x:batch_xs, y:batch_ys, keepratio:0.7})
            # 平均损失
            avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys, keepratio:1.}) / total_batch
        # 显示log
        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x:batch_xs, y:batch_ys, keepratio:1.})
            print("Training accuracy: %.3f" % (train_acc))
        # 保存
        if epoch % save_step == 0:
            saver.save(sess, "save/nets/cnn_mnist_basic.ckpt-" + str(epoch))
    print("FINISH TRAIN")
    
if do_train == 0:
    epoch = training_epochs - 1
    # 加载
    saver.restore(sess, "save/nets/cnn_mnist_basic.ckpt-" + str(epoch))
    # 测试，并输出准确率
    test_acc = sess.run(accr, feed_dict={x:testimg, y:testlabel, keepratio:1.})
    print("Test accuracy: %.3f" % (test_acc))
    print("FINISH TEST")
    
print("ALL FINISH")
        
