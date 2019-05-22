# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:27:15 2019

@author: 19056
"""

#假设有一组数据集 其x和y的对应关系是y约等于2x
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize":[],"loss":[]} #存放批次值和损失值

#生成模拟数据
train_X = np.linspace(-1,1,100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x 但是加入了噪声 np.random.randn(*train_X.shape) 等价于np.random.randn(100)
#显示模拟数据点
plt.plot(train_X , train_Y , 'ro' , label = 'Original data')
plt.legend()
plt.show()

#创建模型
#占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
#模型参数
W = tf.Variable(tf.random_normal([1]),name = "weight")
b = tf.Variable(tf.zeros([0]),name = "bias")

#前向结构
z = tf.multiply(X,W) + b

#反向优化
cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Cradient descent 梯度下降

#初始化变量
init = tf.global_variables_initializer()
#训练参数
training_epochs = 20
display_step = 2

#启动session
with tf.Session() as sess:
    sess.run(init)
    
    #Fit all training data 向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        
        #显示训练的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict ={X:train_X,Y:train_Y})
            print ("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print ("Finish!")
    print ("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))
    #print ("cost:",cost.eval({X:train_X,Y:train_Y}))







