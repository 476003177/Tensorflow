#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#——————————————————导入数据——————————————————————
f=open('data/dataset_1.csv')  
df=pd.read_csv(f)     #读入股票数据
data=np.array(df['max'])  #获取最高价序列
data=data[::-1]      #使用分片反转，使数据按照日期先后顺序排列，[::-1]代表从后向前取值，每次步进值为1

#以折线图展示data
#plt.figure()
#plt.plot(data)
#plt.show()
normalize_data=(data-np.mean(data))/np.std(data)  #标准化，numpy.mean()求取均值，numpy.std()计算全局标准差  
normalize_data=normalize_data[:,np.newaxis]       #增加维度


#生成训练集
#设置常量
time_step=20      #时间步
rnn_unit=10       #hidden layer units
batch_size=60     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率
train_x,train_y=[],[]   #训练集
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

#——————————————————定义神经网络变量——————————————————
#tf.placeholder(dtype, shape=None, name=None):占位符
	#dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
	#shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
	#name：名称，默认是None
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #每批次tensor对应的标签
#输入层、输出层权重、偏置
#tf.Variable(initializer,name)：使用tensorflow在默认的图中创建节点，这个节点是一个变量
	#initializer：初始化参数，可以有tf.random_normal，tf.constant，tf.constant等
	#name：变量的名字
#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)：从正态分布中输出随机值
	#shape: 一维的张量，也是输出的张量
	#mean: 正态分布的均值
    #stddev: 正态分布的标准差
    #dtype: 输出的类型。
    #seed: 一个整数，当设置之后，每次生成的随机数都一样
    #name: 操作的名字
#tf.constant(value, dtype=None, shape=None, name='Const')：创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }

#——————————————————定义神经网络变量——————————————————
#tf.reshape(tensor, shape, name=None)：将tensor变换为参数shape的形式
	#列表中可以存在-1。-1代表的含义是自动计算这一维的大小
#tf.matmul：矩阵乘法；tf.multiply：矩阵点乘
#tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
	#n_hidden：神经元的个数
	#forget_bias：LSTM门的忘记系数，如果等于1，不会忘记任何信息
	#state_is_tuple：默认为True，返回tuple:(c=[batch_size, num_units], h=[batch_size,num_units])
#cell.zero_state(batch_size, dtype)：生成初始化网络的state
#tf.nn.dynamic_rnn(cell,inputs,sequence_length=None,initial_state=None,dtype=None,parallel_iterations=None,swap_memory=False,time_major=False,scope=None)
	#通过inputs中的max_time将网络按时间展开
	#inputs:[batch_size, max_time, size]
	#返回：outputs,last_states
	#outputs：每一个迭代隐状态的输出
	#last_states：由(c,h)组成的tuple
def lstm(batch):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])   #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out #整个网络的输出
    return pred,final_states

#——————————————————训练模型——————————————————
#with:适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，比如文件使用后自动关闭、线程中锁的自动获取和释放等。
#tf.variable_scope(name_or_scope,default_name=None,values=None,initializer=None,regularizer=None,caching_device=None,partitioner=None,custom_getter=None,reuse=None,dtype=None)
	#返回一个用于定义创建variable（层）的op的上下文管理器
#tf.train.AdamOptimizer：实现了Adam算法的优化器 
#tf.train.Saver()：创建一个Saver 来管理模型中的所有变量
	#如果不给tf.train.Saver() 传入任何参数，那么server 将处理graph 中的所有变量。其中每一个变量都以变量创建时传入的名称被保存。
#tf.global_variables()：获得tf定义的全局变量
#tf.(fetches,feed_dict=None,options=None,run_metadata=None)：执行 fetches 中的操作，计算 fetches 中的张量值，用相关的输入变量替换feed_dict中的值
	#返回值：fetches的执行结果
def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(batch_size)
	#以预测值和标签值之间的均方误差作为损失
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))  #损失函数
    #训练过程是最小化loss，lr为学习率
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #将所有图变量进行集体初始化
        #重复训练x次，此处为10000次
        for i in range(10000): #We can increase the number of iterations to gain better result.
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #每10步保存一次参数
                if step%60==0:
                    print("Number of iterations:",i," loss:",loss_)
                    print("model_save",saver.save(sess,'save/stock_predict_1/modle.ckpt'))
                    #I run the code in windows 10,so use  'model_save1\\modle.ckpt'
                    #if you run it in Linux,please use  'model_save1/modle.ckpt'
                step+=1
        print("The train has finished")
train_lstm()

#————————————————预测模型————————————————————
def prediction():
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(1)    #预测时只输入[1,time_step,input_size]的测试数据
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
		#参数恢复
        saver.restore(sess, 'save/stock_predict_1/modle.ckpt') 
        #I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        #if you run it in Linux,please use  'model_save1/modle.ckpt'
		#取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq=train_x[-1]
        predict=[]
		#得到之后100个预测结果
        for i in range(100):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
			#每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
        #以折线图表示结果
        print("The prediction has finished")
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()
        
prediction() 
