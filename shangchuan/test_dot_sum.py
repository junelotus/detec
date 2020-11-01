import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np

gt_loss = [[0,0,0],[1,1,1],[0,0,0]]
pred_loss = [[1,2,3,4],[3,4,5,6],[6,7,8,9]]
print(pred_loss)
#intersect = np.array(gt_loss)*np.array(pred_loss)# np.dot(gt_loss,pred_loss)
d = np.sum(np.array(gt_loss))
print(tf.reduce_sum(pred_loss,axis=-1))
#print(d)
#print(intersect)
