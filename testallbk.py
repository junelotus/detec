import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np
import cv2
batch_size  = 1
img_to_del = None
def read_jpg_gt(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 1)
    return img
def load_img_label(path):
    print(path)
    img = read_jpg_gt(path)
    img = tf.image.resize(img,(64,64))
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img


def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(20000)

#showImg('name',cv2.imread('./score_map/1.jpg'))
score_label = glob.glob('./score_map/*jpg')
label_score_img = tf.data.Dataset.from_tensor_slices(score_label)
label_score_data = label_score_img.map(load_img_label,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)
for score in label_score_data:
	#print(score)
	break
def read_jpg(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 3)
    return img

def load_img_train(path):
    img = read_jpg(path)
    img = tf.image.resize(img,(512,512)) # resize 会使得在后续显示的时候不是none none
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img

# def load_img_train(self):
#         img = tf.image.resize(img_to_del,(512,512)) # resize 会使得在后续显示的时候不是none none
#         img = tf.cast(img, tf.float32)/127.5 - 1 #img/127.5-1
#         return img
def jifentu(img):
	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return np.sum(img)
def to_get_begin_end_time(path):
	img_ori = glob.glob(path)#('./1_ch4_training_images/img_1.jpg')#
	batch_size = 1
	train_img = tf.data.Dataset.from_tensor_slices(img_ori)
	batch_size = 1
	train_data = train_img.map(load_img_train,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)#.shuffle(buffer_size)
	
	for t in train_data:
		pred_score,angle_map,geo_map = net(t,training=False)
		print(geo_map.shape[0])
		size_t =64
		pred_img = np.array(tf.reshape(pred_score,(64,64)))
		shape = geo_map.shape
		t = np.array(tf.reshape(geo_map,shape))
		t = tf.squeeze(t,axis=0)
		t = tf.squeeze(t,axis=-1)
		shape = t.shape[1]
		# print('t.shape',t[1,1,:].shape)
		# print('shape',shape)
		list_cor = []

		for i in  range(shape):
			for j in range(shape):
				if  np.sum(np.array(t[i,j,:])) > 0.0:
					# print('i,j',np.sum(np.array(t[i,j,:])))#np.array(t[i,j,:]))
					position = np.array(t[i,j,:])
					# if len(list_cor) == 0 or (list_cor[-1].all()!=position.all()):
					list_cor.append(position)

		img = np.array(tf.reshape(t[:,:,0],(size_t,size_t)))

		list_cor = list_cor
		# print(list_cor)
		
		img = cv2.imread(path)
		img = cv2.resize(img, (1280, 720))
		r1 = 4#*int(img.shape[1]/1280)
		r2 = 4#*int(img.shape[0]/720) 
		for position in list_cor:
			cv2.line(img,(int(position[0]* r2),int(position[1]* r1)),(int(position[2]* r2),int(position[3]* r1)),(0,255,0),1)
			cv2.line(img,(int(position[2]* r2),int(position[3]* r1)),(int(position[4]* r2),int(position[5]* r1)),(255,255,0),1)
			cv2.line(img,(int(position[4]* r2),int(position[5]* r1)),(int(position[6]* r2),int(position[7]* r1)),(0,255,255),1)
			cv2.line(img,(int(position[6]* r2),int(position[7]* r1)),(int(position[0]* r2),int(position[1]* r1)),(0,0,255),1)
		#showImg('ori',img)#cv2.imread('geo_map/1_0.jpg'))
		cv2.imwrite('./first.jpg',img)
		# print(pred_img)
		pred_img = cv2.resize(pred_img, (1280, 720))
		pred_img = pred_img*255
		allwhite = jifentu(pred_img)
		# cv2.imwrite('./second.jpg',pred_img*255)
		return allwhite
net = tf.keras.models.load_model('net.h5')
path  = './jpg15/danmu2.jpg'#'./1_ch4_training_images/img_3.jpg'#
path = glob.glob('./192423.mp4')





def getAll(path):
	for p in path:
		capture = cv2.VideoCapture(p)
		while capture.isOpened():
			timestamp = int(capture.get(cv2.CAP_PROP_POS_MSEC))
			frame_exists, img_to_del = capture.read()
			if  not frame_exists:
				print('because end')
				break
			cv2.imencode('.jpg', img_to_del)[1].tofile('./now_jpg/1.jpg')
			allwhite = to_get_begin_end_time('./now_jpg/1.jpg')
			print('allwhite',allwhite)
getAll(path)