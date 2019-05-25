# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:22:04 2019

@author: 19056
"""

#get_variable和variable_scope一起使用
import tensorflow as tf

tf.reset_default_graph()

#var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
#var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32) 

with tf.variable_scope("test1"):
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    with tf.variable_scope("test3"):
        var3 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    
with tf.variable_scope("test2"):
    var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)

print ("var1:",var1.name)
print ("var2:",var2.name)
print ("var3:",var3.name)

with tf.variable_scope("test1",reuse=True):
    var4 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    with tf.variable_scope("test3"):
        var5 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
        
print ("var4:",var4.name)
print ("var5:",var5.name)









