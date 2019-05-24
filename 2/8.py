# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:19:30 2019

@author: 19056
"""

#按照训练时间记录中间点的检查点 trainMonit
import tensorflow as tf
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step,1)

with tf.train.MonitoredTrainingSession(checkpoint_dir="log/checkpoints",save_checkpoint_secs = 2) as sess:
    print (sess.run([global_step]))
    while not sess.should_stop():
        i = sess.run(step)
        print (i)
        
