# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:01:47 2019

@author: 19056
"""

#假设有一组数据集 其x和y的对应关系是y约等于2x
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize":[],"loss":[]} #存放批次值和损失值
def moving_average(a,w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]


#生成模拟数据
train_X = np.linspace(-1,1,100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x 但是加入了噪声 np.random.randn(*train_X.shape) 等价于np.random.randn(100)
#显示模拟数据点
plt.plot(train_X , train_Y , 'ro' , label = 'Original data')
plt.legend()
plt.show()

#创建模型
#占位符
#X = tf.compat.v1.placeholder("float")
#Y = tf.compat.v1.placeholder("float")
inputdict = {
    'x':tf.compat.v1.placeholder("float"),
    'y':tf.compat.v1.placeholder("float")        
}

#模型参数
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

#前向结构
z = tf.multiply(inputdict['x'], W)+ b

#反向优化
cost = tf.reduce_mean(tf.square(inputdict['y']-z))
learning_rate = 0.01
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Cradient descent 梯度下降

#初始化变量
init = tf.compat.v1.global_variables_initializer()
#训练参数
training_epochs = 20 #迭代的次数
display_step = 2

#启动session
with tf.compat.v1.Session() as sess:#sess.run() 网络节点的运算
    sess.run(init)
    
    #Fit all training data 向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={inputdict['x']:x,inputdict['y']:y}) #feed机制将真实的数据灌输到占位符上
        
        #显示训练的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict ={inputdict['x']:train_X,inputdict['y']:train_Y})
            print ("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print ("Finish!")
    print ("cost=",sess.run(cost,feed_dict={inputdict['x']:train_X,inputdict['y']:train_Y}),"W=",sess.run(W),"b=",sess.run(b))
    #print ("cost:",cost.eval({X:train_X,Y:train_Y}))
    
    #图形显示
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    
    plt.show()
    
    print ("x=0.2 , z=",sess.run(z,feed_dict={inputdict['x']:0.2}))
    







