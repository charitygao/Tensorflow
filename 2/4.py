# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:36:43 2019

@author: 19056
"""

#使用注入机制获取节点 fetch
import tensorflow as tf
a = tf.compat.v1.placeholder(tf.int16)
b = tf.compat.v1.placeholder(tf.int16)
add = tf.add(a,b)
mul = tf.multiply(a,b) #a与b相乘
with tf.Session() as sess:
    #计算具体的数值
    print ("相加：%i" % sess.run(add,feed_dict={a:3,b:4}))
    print ("相乘：%i" % sess.run(mul,feed_dict={a:3,b:4}))
    print (sess.run([add,mul],feed_dict={a:3,b:4}))
