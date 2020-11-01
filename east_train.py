import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np

img_ori = glob.glob('./1_ch4_training_images/*jpg')
label_ori = glob.glob('./2_ch4_training_localization_transcription_gt/*txt')
train_a = tf.data.Dataset.from_tensor_slices(img_ori)

def read_jpg(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 3)
    return img

def load_img(path):
    img = read_jpg(path)
    img = tf.image.resize(img,(512,512)) # resize 会使得在后续显示的时候不是none none
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img

buffer_size  = 20
train_a = train_a.map(load_img,num_parallel_calls = AUTOTUNE).cache().shuffle(buffer_size).batch(1)

def getVGG16Pooling2ToPooling2():
    model = keras.Sequential()
    # model.add(keras.layers.MaxPooling2D((2,2),strides = 2,padding = 'same'))

    model.add(keras.layers.Conv2D(256,3,strides = 1,padding = 'same',activation = 'relu'))
    model.add(keras.layers.Conv2D(256,3,strides = 1,padding = 'same',activation = 'relu'))
    model.add(keras.layers.Conv2D(256,3,strides = 1,padding = 'same',activation = 'relu'))
    # model.add(keras.layers.MaxPooling2D((2,2),strides = 2,padding = 'same'))

    model.add(keras.layers.Conv2D(512,3,strides = 1,padding = 'same',activation = 'relu'))
    model.add(keras.layers.Conv2D(512,3,strides = 1,padding = 'same',activation = 'relu'))
    model.add(keras.layers.Conv2D(512,3,strides = 1,padding = 'same',activation = 'relu'))
    # model.add(keras.layers.MaxPooling2D((2,2),strides = 2,padding = 'same'))

    model.add(keras.layers.Conv2D(512,3,strides = 1,padding = 'same',activation = 'relu'))
    model.add(keras.layers.Conv2D(512,3,strides = 1,padding = 'same',activation = 'relu'))
    model.add(keras.layers.Conv2D(512,3,strides = 1,padding = 'same',activation = 'relu'))
    # model.add(keras.layers.MaxPooling2D((2,2),strides = 2,padding = 'same'))
    return model

def downSample(filter,size,padding ='same'):
    result  = keras.Sequential()
    result.add(keras.layers.Conv2D(filter,size,strides = 2,padding=padding))
    return result

def upSample(filter,size):
    result  = keras.Sequential()
    # result.add(keras.layers.Unpooling((2,2),strides = 2,padding = 'same'))
    # result.add(keras.layers.MaxPooling2D((2,2),strides = 0.5,padding = 'same'))
    result.add(keras.layers.Conv2DTranspose(filter,size,strides = 2,padding='same'))
    return result

def conv3conv1Sample(filter,size1,size2):
    result  = keras.Sequential()
    result.add(keras.layers.Conv2D(filter,size1,strides = 1,padding='same'))
    result.add(keras.layers.Conv2D(filter,size2,strides = 1,padding='same'))
    return  result

def afterVGG16():
    inputs = keras.layers.Input(shape=[512,512,3])
    model_vgg16 = getVGG16Pooling2ToPooling2()
    x = model_vgg16(inputs)
    
    x = keras.layers.Conv2D(16,7,strides = 2 ,padding='same')(x)
    print('first',x.shape)
    # Conv2D(filters, kernel_size, strides=(1, 1),...
    # model.add(keras.layers.Conv2D(64,3,strides = 2,padding = 'same',activation = 'relu'))
    # model.add(keras.layers.Conv2D(128,3,strides = 2,padding = 'same',activation = 'relu'))
    # model.add(keras.layers.Conv2D(256,3,strides = 2,padding = 'same',activation = 'relu'))
    # model.add(keras.layers.Conv2D(384,3,strides = 2,padding = 'same',activation = 'relu'))
    downsample_list = [
        downSample(64,4),
        downSample(128,4),
        downSample(256,4),
        downSample(384,4),
    ]

    downresult_list = []

    i = 0
    for down in downsample_list:
        # print('junejune')
        x = down(x)
        # print('junejune1')
        downresult_list.append(x)
        print('down',i,x.shape)
        i = i+1

    downresult_list = reversed(downresult_list[:-1])#不要最后一层进行反转list

    upsample_list = [
        upSample(256,3),
        upSample(128,3),
        upSample(64,3),
        # upSample(384,3)
    ]

    conv3conv1_list = [
        conv3conv1Sample(128,1,3),
        conv3conv1Sample(64,1,3),
        conv3conv1Sample(32,1,3),
    ]
    i = 0
    for up,inputs_from_down ,conv3conv1 in zip(upsample_list,downresult_list,conv3conv1_list):
        x = up(x)
        # print('up',i,x.shape)
        x = keras.layers.Concatenate()([x,inputs_from_down])
        # print('junejune22')
        x = conv3conv1 (x)
        print('up',i,x.shape)
        i = i+1

    pred_score = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    angle_map = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map = keras.layers.Conv2D(4,1,strides=1,padding = 'same')(x)

    return keras.Model(inputs = inputs,outputs = [pred_score,angle_map,geo_map] )

def lossScoreMap(gt_score,pred_score):
    beta = 1 - np.sum(gt_score)/(gt.size)
    score_loss_bacth = beta*gt_score*np.log(pred_score) + (1-beta)*(1-gt_score)*(1-pred_score)
    mean = np.mean(score_loss_bacth)
    return mean


def scoreMapLossDice(gt_loss,pred_loss,axis=(1,2,3),smooth=1e-5):
    inse = tf.reduce_sum(gt_loss*gt_loss,axis=axis)
    l = tf.reduce_sum(gt_loss,axis=axis)
    r = tf.reduce_sum(pred_loss,axis=axis)
    return 1-(2.0*inse+smooth)/(l+r+smooth)

def generateScoreMapGT(gt_batch):
    # 根据gt.txt来生成gt_score gt_angle gt_geo


    return gt_score,gt_angle,gt_geo

# def geoLoss(gt_geo,pred_geo):
    

    # batch_size = gt_loss.size()


    # intersect = np.sum(np.array(gt_loss)*np.array(pred_loss))
    # union = np.sum(np.array(gt_loss)) + np.sum(np.array(pred_score))
    # return 1 -  2*intersect / union


model  = afterVGG16()
# model = getVGG16Pooling2ToPooling2()
# print(model)