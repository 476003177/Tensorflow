# coding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
import input_data

# 加载mnist数据
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels
ntrain = trainimgs.shape[0]  # 查看维度
ntest = testimgs.shape[0]
dim = trainimgs.shape[1]
nclasses = trainlabels.shape[1]
print("MNIST loaded")

diminput = 28  # mnist图象数据为28*28，要变成一维的28
dimhidden = 128  # 隐层有128个神经元
dimoutput = nclasses  # 要分成的类别
nsteps = 28  # 28*28，变成一维28后要输入28次，即有28步
weights = {
    # 输入为一维的28，有128个神经元
    'hidden':tf.Variable(tf.random_normal([diminput, dimhidden])),
    # 128个神经元输出为类别
    'out':tf.Variable(tf.random_normal([dimhidden, dimoutput]))
    }
biases = {
    'hidden':tf.Variable(tf.random_normal([dimhidden])),
    'out':tf.Variable(tf.random_normal([dimoutput]))
    }


# 有未知错误
def _RNN(_X, _W, _b, _nsteps, _name):
    # 输入RNN的数据为[batchsize,nsteps,diminput]
    # 要将输入数据降维成[nsteps*batchsize,diminput]
    # 先将batchsize和nsteps维度置换，再将其相乘
    _X = tf.transpose(_X, [1, 0, 2])
    _X = tf.reshape(_X, [-1, diminput])
    # 隐层输入，先整体计算再切分各个batch
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 切分batch
    _Hsplit = tf.split(_H, _nsteps, axis=0)
    # LSTM单元
    # 利用with指定命名域，避免再次运行冲突
    with tf.variable_scope(_name) as scope:
        scope.reuse_variables()  # 变量共享，当命名域冲突时候，各自的变量互相共享
        # 输入神经元为dimhidden(128)，输出也要dimhidden(128),forget_bias=1.0即不遗忘
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        # LSTM两个输出(本次结果和传到下次的结果)
        _LSTM_0, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, dtype=tf.float32)
    # 输出，只要最后一层LSTM的输出，并接上分类神经元
    _O = tf.matmul(_LSTM_0[-1], _W['out']) + _b['out']
    # 返回
    return{
        'X':_X, 'H':_H, 'Hsplit':_Hsplit,
        'LSTM_0':_LSTM_0, 'LSTM_S':_LSTM_S, 'O':_O
        }


learning_rate = 0.001
x = tf.placeholder("float", [None, nsteps, diminput])
y = tf.placeholder("float", [None, dimoutput])
# RNN输出，指定命名为basic，避免重复运行
myrnn = _RNN(x, weights, biases, nsteps, 'basic')
# 预测值
pred = myrnn['O']
# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 优化器
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 准确度
accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
# 初始化
init = tf.global_variables_initializer()

training_epochs = 5  # 重复次数
batch_size = 16
display_step = 1
sess = tf.Session()
sess.run(init)
print("Start optimization")
for epoch in range(training_epochs):
        avg_cost = 0;
        # batch数目
        # total_batch=int(mnist.train._num_examples/batch_size)
        total_batch = 100
        for i in range(total_batch):
            # 获取该batch的数据，batch_xs为数据，batch_ys为标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 整理格式，升维，以便输入网络
            batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
            # 运行
            feeds = {x:batch_xs, y:batch_ys}
            sess.run(optm, feed_dict=feeds)
            # 平均损失
            avg_cost += sess.run(cost, feed_dict=feeds) / total_batch
        # 显示log
        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            feeds = {x:batch_xs, y:batch_ys}
            train_acc = sess.run(accr, feed_dict=feeds)
            print("Training accuracy: %.3f" % (train_acc))
            testimgs = testimgs.reshape((ntest, nsteps, diminput))
            feeds = {x:testimgs, y:testlabels, istate:np.zeros((ntest, 2 * dimhidden))}
            test_acc = sess.run(accr, feed_dict=feeds)
            print("Test accuracy: %.3f" % (test_acc))
#         # 保存
#         if epoch % save_step == 0:
#             saver.save(sess, "save/nets/cnn_mnist_basic.ckpt-" + str(epoch))
print("FINISH")
