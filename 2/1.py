# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:11:02 2019

@author: 19056
"""

#编写hello world程序演示session的使用
import tensorflow as tf
hello = tf.constant('Hello TensorFlow') #定义一个常量
#print (hello)
sess = tf.compat.v1.Session() #建立一个session
print (sess.run(hello)) #通过session里的run来运行结果
sess.close() #关闭session