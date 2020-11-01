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
# def jifentu(img):
# 	shape = img.shape
# 	for i in range 
# 	for i in range(shape[0]):
# 		for j in range(shape[1]):
# 			img[i][j] = img[i][j]+img[i-1][j-1]

def read_jpg_gt(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 1)
    return img
def load_img_label(path):
    #print(path)
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
	##print(score)
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
path  ='danmu4.jpg'#'./jpg17/IMG_20200709_110101.jpg'#'test_ocr.jpg'#'./jpg17/IMG_20200709_110101.jpg'#'./jpg15/danmu2.jpg'# 'join.zego.jpg'#
img_ori_to_get_shape = cv2.imread(path).shape
img_ori = glob.glob(path)#('./1_ch4_training_images/img_1.jpg')#
batch_size  = 1
train_img = tf.data.Dataset.from_tensor_slices(img_ori)
batch_size = 1
train_data = train_img.map(load_img_train,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)#.shuffle(buffer_size)
net = tf.keras.models.load_model('net.h5')
for t in train_data:
	pred_score,angle_map,geo_map = net(t,training=False)
	#print(geo_map.shape[0])
	size_t =64
	pred_img = np.array(tf.reshape(pred_score,(64,64)))
	shape = geo_map.shape
	t = np.array(tf.reshape(geo_map,shape))
	t = tf.squeeze(t,axis=0)
	t = tf.squeeze(t,axis=-1)
	shape = t.shape[1]
	#print('t.shape',t[1,1,:].shape)
	#print('shape',shape)
	list_cor = []

	for i in  range(shape):
		for j in range(shape):
			if  np.sum(np.array(t[i,j,:])) > 0.0:
				#print('i,j',np.sum(np.array(t[i,j,:])))#np.array(t[i,j,:]))
				position = np.array(t[i,j,:])
				# if len(list_cor) == 0 or (list_cor[-1].all()!=position.all()):
				list_cor.append(position)

	img = np.array(tf.reshape(t[:,:,0],(size_t,size_t)))
	'''
	x1  = np.array(tf.reshape(t[:,:,0],(size_t,size_t)))
	y1  = np.array(tf.reshape(t[:,:,1],(size_t,size_t)))
	x2  = np.array(tf.reshape(t[:,:,2],(size_t,size_t)))
	y2  = np.array(tf.reshape(t[:,:,3],(size_t,size_t)))
	x3  = np.array(tf.reshape(t[:,:,4],(size_t,size_t)))
	y3  = np.array(tf.reshape(t[:,:,5],(size_t,size_t)))
	x4  = np.array(tf.reshape(t[:,:,6],(size_t,size_t)))
	y4  = np.array(tf.reshape(t[:,:,7],(size_t,size_t)))
	shape = x1.shape
	for i in  range(shape[0]):
		for j in range(shape(1)):
			if x1[i][j]

	'''

	# #print(img)
#for i in range(size_t):
#	for j in range(size_t):
#		if img[i][j] >=255:
#			##print(img[i][j])
#			img[i][j] = 255
#		else :
#			img[i][j] = 0

	list_cor = list_cor
	#print(list_cor)

	img = cv2.imread(path)
	img = cv2.resize(img, (img_ori_to_get_shape[1], img_ori_to_get_shape[0]))
	#print('pred_img.shape',pred_img.shape)
	pred_img = cv2.resize(pred_img, (img_ori_to_get_shape[1], img_ori_to_get_shape[0]))
	#print('pred_img.shape',pred_img.shape)
	# jifentu
	# pred_img = cv2.resize(pred_img, (1280, 720))*255
	# pred_img = cv2.cvtColor(pred_img.astype(np.int8), cv2.COLOR_RGB2GRAY)
	_, pred_img = cv2.threshold(pred_img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	

	contours, _ = cv2.findContours(pred_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2], reverse=True)
	i = 0
	for cnt in contours:
		x, y, w, h =  cv2.boundingRect(cnt)
		imgsub = img[max(y-15,0):min(img_ori_to_get_shape[0],y+h+15),max(x-100,0):min(x+w+100,img_ori_to_get_shape[1])]

		# contourssub, _ = cv2.findContours(imgsub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		# contourssub = sorted(contourssub, key=lambda c: cv2.boundingRect(c)[2], reverse=True)
		imgsub = cv2.bitwise_not(imgsub)
		# imgsub = cv2.cvtColor(imgsub, cv2.COLOR_RGB2GRAY)
		# _, imgsub = cv2.threshold(imgsub.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		print(imgsub.shape)
		# if  imgsub.shape.any()==0:
		cv2.imencode('.jpg', imgsub)[1].tofile('danmu/'+str(i)+'.jpg')
		cv2.rectangle(img, pt1=(x-15,y-15), pt2=(x+w+15,y+h+15),color=(0, 0, 255), thickness=2)
		
		i = i+1
	r1 = int(img.shape[1]/1280*10) #10 #*int(img.shape[1]/1280)
	r2 = int(img.shape[0]/720*10) #10 #*int(img.shape[0]/720)
	# for position in list_cor:
	# 	# 根据其中的点数多少来进行极大值抑制
	# 	cv2.line(img,(int(position[0]* r2),int(position[1]* r1)),(int(position[2]* r2),int(position[3]* r1)),(0,255,0),1)
	# 	cv2.line(img,(int(position[2]* r2),int(position[3]* r1)),(int(position[4]* r2),int(position[5]* r1)),(255,0,0),1)
	# 	cv2.line(img,(int(position[4]* r2),int(position[5]* r1)),(int(position[6]* r2),int(position[7]* r1)),(0,0,255),1)
	# 	cv2.line(img,(int(position[6]* r2),int(position[7]* r1)),(int(position[0]* r2),int(position[1]* r1)),(0,0,255),1)
	
		# p1 = cv2.Point(int(position[0]* r2),int(position[1]* r1))
		# p2 = cv2.Point(int(position[2]* r2),int(position[3]* r1))
		# p3 = cv2.Point(int(position[4]* r2),int(position[5]* r1))
		# p4 = cv2.Point(int(position[6]* r2),int(position[7]* r1))
		# cv2.line(pred_img,(p1,p2),(0,255,0),1)
		# cv2.line(pred_img,(p2,p3),(255,0,0),1)
		# cv2.line(pred_img,(p3,p4),(0,0,255),1)
		# cv2.line(pred_img,(p4,p1),(0,0,255),1)

		#showImg('ori',img)#cv2.imread('geo_map/1_0.jpg'))
	img = cv2.resize(img, (img_ori_to_get_shape[1], img_ori_to_get_shape[0]))
	cv2.imwrite('./first.jpg',img)
	# #print(pred_img)
	# pred_img = cv2.resize(pred_img, (4160, 3120))
	
	cv2.imwrite('./second.jpg',pred_img)#*255)
	# for i in range()
	# after_resize = cv2.resize(pred_img*255,(4160,3120))
	# img_ori = cv2.imread(path)
	# for i in range(3120-1):
	# 	for j  in range(4160-1):
	# 		if after_resize[i][j] == 255:
	# 			img_ori[i][j] =0
	# showImg('second.jpg',img_ori)#after_resize)#pred_img*255)
	#showImg('first',img)
	#showImg('ori',t)


