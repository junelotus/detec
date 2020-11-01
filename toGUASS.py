import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np
import cv2


img = cv2.imread('~/Download/blured.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img = cv2.GaussianBlur(img,(5,5),0)
