# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:20:21 2019

@author: 19056
"""
#with session的使用
import tensorflow as tf
a = tf.constant(3) #定义常量3
b = tf.constant(4) #定义常量4
with tf.Session() as sess: #建立session
    print ("相加：%i" % sess.run(a+b))
    print ("相乘：%i" % sess.run(a*b))