#!/usr/bin/env python
# coding: utf-8

# In[5]:


#     data_int = []
#     for elem in data:
#         for sub in elem:
#             print('sub',int(sub))
#     print(data[:,:-1])#.type)

#     file = tf.io.read_file(path)
#     file = tf.io.decode_csv(file,record_defaults='int32')
#     print(file)
#     for line in file:
    
#         print(line)

# filename_queue = tf.train.string_input_producer([“file0.csv”, “file1.csv”])
# #每次一行
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
# #解析每次的一行，默认以逗号分开
# image_v, label = tf.decode_csv(
# value, record_defaults=record_defaults)
# result.label = tf.string_to_number(label,tf.int32)
# image= tf.string_to_number(image_v,tf.int32)
# image = tf.reshape(image,[result.depth, result.height, result.width])
# //Convert from [depth, height, width] to [height, width, depth].
# result.uint8image = tf.transpose(image, [1, 2, 0])123456789101112


# In[6]:


# def drawPictureUseCornerDotOri(AffineImg,img_to_correct,width,height,canvas_point):
#     u_k = (canvas_point[3][1]-canvas_point[0][1])/(canvas_point[3][0]-canvas_point[0][0])
#     u_d = canvas_point[3][1] - u_k*canvas_point[3][0]
    
#     b_k = (canvas_point[2][1]-canvas_point[1][1])/(canvas_point[2][0]-canvas_point[1][0])
#     b_d = canvas_point[2][1] - b_k*canvas_point[2][0]
    
#     l_k = (canvas_point[1][1]-canvas_point[0][1])/(canvas_point[1][0]-canvas_point[0][0])
#     l_d = canvas_point[0][1] - l_k*canvas_point[0][0]

#     r_k =  (canvas_point[2][1]-canvas_point[3][1])/(canvas_point[2][0]-canvas_point[3][0])
#     r_d = canvas_point[2][1] - r_k*canvas_point[2][0]
    
#     score_map =  np.ones((height,width,3),dtype=np.uint8)#img_to_correct[0:height,0:width]
   
    
#     for x in range(img_to_correct.shape[0]):
#         for y in range(img_to_correct.shape[1]):
#             if ( ((u_k >= 0 and y-u_k*x-u_d  <= 0.0 ) or ( u_k< 0 and  y-u_k*x-u_d  >= -1.0))
#              and  ( (b_k >=0 and  y-b_k*x-b_d  >= 0.0) or ( b_k <=0  and y-b_k*x-b_d  <= -1.0)) 
#             and  y-l_k*x-l_d  >= 1.0
#              and y-r_k*x-r_d <= -1.0):
#                    score_map[x][y] = 1 
# #                 src = np.array([x,y,1])
# #                 # ##print('nownownownow')
# #                 # ##print(src)
# #                 dst = np.dot(AffineMatrix,src)
# #                 dst_x = int(dst[0]/dst[-1])
# #                 dst_y = int(dst[1]/dst[-1])
# #                 if dst_x >= 0 and dst_x < result_perspective.shape[0] and dst_y >=0 and dst_y < result_perspective.shape[1]:
# #                     result_perspective[dst_x][dst_y] = img_to_correct[x][y]
# #     cv2.imshow('score_map',score_map)
#     return score_map


# In[7]:


# def findAngle(line,angle_gt_sub):
#     # 找到最低点，和右上角的点
#     flag  = 0
#     i = 0
#     max_y = -1
#     for point in line:
#         t = point[1]
# #         if t == max_y:
# #             angle = 0
# #             break
#         if t >= max_y:
#             flag = i
#             max_y = t
#         i = i+1
            
#     second = (flag+3)%4
#     print(second,flag)
#     if (line[flag][0]-line[second][0]) is 0:
#         angle = 0 
#     else:
#         angle = math.atan((line[flag][1]-line[second][1])/(line[flag][0]-line[second][0]))*180
#     print(angle)
#     return line
        
        


# In[8]:


# w = int(1280/4)
# h = int(720/4)
# for path in label_ori:

#     file = open("./2_ch4_training_localization_transcription_gt/gt_img_268.txt",encoding='UTF-8-sig')
#     while True:
#         line = file.readline()
#         if not line:
#             break
#         line = np.array(line.split(',')[0:8])
#         line = line.astype(np.int32)
#         line = line.reshape(-1,2)
#         line = (line/4).astype(np.int32)
#         #to construct the score_map angle_map geo_map,the result is three vector
#         angle_gt_sub  = np.zeros((w,h,1),dtype=np.float32)
#         findAngle(line,angle_gt_sub)
        
#         print(line)
        
#     file.close()
#     break

# def generateScoreMapGT(gt_batch):
#     # 根据gt.txt来生成gt_score gt_angle gt_geo
    

#     return gt_score,gt_angle,gt_geo
    


# In[3]:


import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.config.gpu.set_per_process_memory_growth(enabled=True)
# tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


# In[2]:


img_ori = glob.glob('./1_ch4_training_images/*jpg')


# In[3]:


# label_ori = glob.glob('./2_ch4_training_localization_transcription_gt/*txt')
score_label = glob.glob('./score_map/*jpg')
angle_label = glob.glob('./angle_map/*jpg')
geo_label_0 = glob.glob('./geo_map/*_0.jpg')
geo_label_1 = glob.glob('./geo_map/*_1.jpg')
geo_label_2 = glob.glob('./geo_map/*_2.jpg')
geo_label_3 = glob.glob('./geo_map/*_3.jpg')
geo_label_4 = glob.glob('./geo_map/*_4.jpg')
geo_label_5 = glob.glob('./geo_map/*_5.jpg')
geo_label_6 = glob.glob('./geo_map/*_6.jpg')
geo_label_7 = glob.glob('./geo_map/*_7.jpg')

def takeFirst(path):
        index = path.split('/')[2].split('.')[0]
        return int(index)

def takeSecond(path):
        index = path.split('/')[2].split('_')[0]
        return int(index)

score_label.sort(key=takeFirst)
angle_label.sort(key=takeFirst)
geo_label_0.sort(key=takeSecond)
geo_label_1.sort(key=takeSecond)
geo_label_2.sort(key=takeSecond)
geo_label_3.sort(key=takeSecond)
geo_label_4.sort(key=takeSecond)
geo_label_5.sort(key=takeSecond)
geo_label_6.sort(key=takeSecond)
geo_label_7.sort(key=takeSecond)
# label_ori


# In[4]:


# score_label


# In[5]:


angle_label


# In[6]:


# geo_label


# In[7]:


geo_label_2


# In[8]:


score_gt = []
angle_gt = []
geo_gt = []


# In[9]:


# img_ori.sort()
# img_ori


# In[10]:


batch_size  = 60
train_img = tf.data.Dataset.from_tensor_slices(img_ori)
label_score_img = tf.data.Dataset.from_tensor_slices(score_label)
label_angle_img = tf.data.Dataset.from_tensor_slices(angle_label)
label_geo_img = tf.data.Dataset.from_tensor_slices((geo_label_0,geo_label_1,geo_label_2,geo_label_3,geo_label_4,geo_label_5,geo_label_6,geo_label_7))


# In[11]:


label_score_img


# In[12]:


label_angle_img


# In[13]:


label_geo_img


# In[14]:


train_img


# In[15]:


def read_jpg(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 3)
    return img


# In[16]:


def read_jpg_gt(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 1)
    return img


# In[17]:


def load_img_train(path):
    img = read_jpg(path)
    
#     index = path.split('/')[2].split('.')[0].split('_')[-1]
#     print(path)
    img = tf.image.resize(img,(512,512)) # resize 会使得在后续显示的时候不是none none
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img


# In[18]:


def load_img_label(path):
    print(path)
    img = read_jpg_gt(path)
    img = tf.image.resize(img,(128,128))
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img


# In[19]:


def load_img_geo_label(path0,path1,path2,path3,path4,path5,path6,path7):
    img_0 = read_jpg_gt(path0)
    img_0 = tf.image.resize(img_0,(128,128)) # resize 会使得在后续显示的时候不是none none
    
    
    img_1 = read_jpg_gt(path1)
    img_1 = tf.image.resize(img_1,(128,128))
    
    
    img_2 = read_jpg_gt(path2)
    img_2 = tf.image.resize(img_2,(128,128))
    
    
    img_3 = read_jpg_gt(path3)
    img_3 = tf.image.resize(img_3,(128,128))
    
    img_4 = read_jpg_gt(path4)
    img_4 = tf.image.resize(img_4,(128,128))
    
    img_5 = read_jpg_gt(path5)
    img_5 = tf.image.resize(img_5,(128,128))
    
    img_6 = read_jpg_gt(path6)
    img_6 = tf.image.resize(img_6,(128,128))
    
    img_7 = read_jpg_gt(path7)
    img_7 = tf.image.resize(img_7,(128,128))
    
    img = tf.stack([img_0, img_1,img_2,img_3,img_4,img_5,img_6,img_7], axis=2)

    return img


# In[20]:


batch_size = 1
train_data = train_img.map(load_img_train,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)#.shuffle(buffer_size)


# In[21]:


buffer_size  = 20

label_score_data = label_score_img.map(load_img_label,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)
label_angle_data = label_angle_img.map(load_img_label,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)


# In[22]:


label_geo_data = label_geo_img.map(load_img_geo_label,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)


# In[23]:


label_score_data


# In[24]:


label_angle_data


# In[25]:


label_geo_data


# In[26]:


train_data


# In[27]:


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


# In[28]:


# train_label = train_label.map(read_txt,num_parallel_calls = AUTOTUNE).cache().shuffle(buffer_size).batch(batch_size)


# In[29]:


def downSample(filter,size,padding ='same'):
    result  = keras.Sequential()
    result.add(keras.layers.Conv2D(filter,size,strides = 2,padding=padding))
    return result


# In[30]:


def upSample(filter,size):
    result  = keras.Sequential()
    # result.add(keras.layers.Unpooling((2,2),strides = 2,padding = 'same'))
    # result.add(keras.layers.MaxPooling2D((2,2),strides = 0.5,padding = 'same'))
    result.add(keras.layers.Conv2DTranspose(filter,size,strides = 2,padding='same'))
    return result


# In[31]:


def conv3conv1Sample(filter,size1,size2):
    result  = keras.Sequential()
    result.add(keras.layers.Conv2D(filter,size1,strides = 1,padding='same'))
    result.add(keras.layers.Conv2D(filter,size2,strides = 1,padding='same'))
    return  result


# In[32]:


def afterVGG16():
    inputs = keras.layers.Input(shape=[512,512,3])
    model_vgg16 = getVGG16Pooling2ToPooling2()
    x = model_vgg16(inputs)
    
    x = keras.layers.Conv2D(16,7,strides = 2 ,padding='same')(x)
    print('first',x.shape)
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

    pred_score = keras.layers.Conv2D(1,1,strides=1,padding = 'same',activation = 'sigmoid')(x)#,activation = 'sigmoid')(x)
#     pred_score = 
    angle_map = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_0 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_1 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_2 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_3 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_4 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_5 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_6 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map_7 = keras.layers.Conv2D(1,1,strides=1,padding = 'same')(x)
    geo_map = tf.stack([geo_map_0, geo_map_1,geo_map_2,geo_map_3,geo_map_4,geo_map_5,geo_map_6,geo_map_7], axis=3)
#     print(geo_map)
    return keras.Model(inputs = inputs,outputs = [pred_score,angle_map,geo_map] )


# In[33]:


# afterVGG16


# In[34]:


def lossScoreMap(gt_score,pred_score,smooth=1e-5):
    batch_size,w,h,_ = gt_score.shape
    # print(gt_score.shape)
    beta = 1 - tf.reduce_sum(gt_score,axis=(-2,-3))/(w*h)
    # beta = 1 -  tf.reduce_sum(gt_score)/(gt_score.size)
    score_loss_bacth = beta*gt_score*tf.math.log(pred_score+smooth) + (1-beta)*(1-gt_score)*tf.math.log(1-pred_score+smooth)
    mean = tf.reduce_mean(score_loss_bacth)
    return mean


# In[35]:


# score loss
# def scoreMapLossDice(gt_loss,pred_loss,axis=(1,2,3),smooth=1e-5):
    
#     inse = tf.reduce_sum(gt_loss*pred_loss)
#     l = tf.reduce_sum(gt_loss,axis=axis)
#     r = tf.reduce_sum(pred_loss,axis=axis)
#     return -tf.math.log((2.0*inse+smooth)/(l+r+smooth))
#     # intersection = tf.reduce_sum(prediction * target, axis=axis)
#     # p = tf.reduce_sum(prediction, axis=axis)
#     # t = tf.reduce_sum(target, axis=axis)
#     # numerator = tf.reduce_mean(intersection + smooth)
#     # denominator = tf.reduce_mean(t + p + smooth)
#     # dice_loss = -tf.math.log(2.*numerator) + tf.math.log(denominator)

#     # return dice_loss
loss_obj =keras.losses.BinaryCrossentropy(from_logits=True)
def scoreMapLossDice(gt_loss,pred_loss,smooth=1e-5):
    
    batch_size,w,h,c =  gt_loss.shape
    
    gt_loss = tf.reshape(gt_loss,(batch_size,w*h*c))
    pred_loss = tf.reshape(pred_loss,(batch_size,w*h*c))

    
    intersection = tf.reduce_sum(gt_loss*pred_loss,axis=-1)#
    l = tf.reduce_sum(gt_loss,axis=-1)
    r = tf.reduce_sum(pred_loss,axis=-1)
    coeff = (2*intersection+1)/(l+r+1)

    dice_loss = 1.-tf.reduce_sum(coeff)/batch_size 
    return dice_loss
    


# In[36]:


# angle loss
def lossAngle(gt_angle,pred_angle):
    return tf.reduce_mean(1-tf.cos(gt_angle-pred_angle))


# In[37]:


# geo map L1 loss
def lossGeo(labels,predictions):#,scope=tf.GraphKeys.LOSSES):
    # with tf.variable_scope(scope):
        batch_size,w,h,_,_ = labels.shape
        diff = tf.abs(labels-predictions)
        less_than_one=tf.cast(tf.less(diff,1.0),tf.float32)   #Bool to float32
        smooth_l1_loss=(less_than_one*0.5*diff**2)+(1.0-less_than_one)*(diff-0.5)#同上图公式
        # print('smooth_l1_loss',smooth_l1_loss.shape)
        smooth_l1_loss= tf.reduce_sum(smooth_l1_loss,axis=-2)# = tf.add(smooth_l1_loss,axis=-2)
        return tf.math.log(tf.reduce_mean(tf.reshape(smooth_l1_loss,(batch_size,w,h,1))))#tf.reduce_mean(smooth_l1_loss) #取平均值
    #tf.math.log(
    # batch_size,w,h,_,_ = gt_geo.shape
    # gt_geo = tf.reshape(gt_geo,(batch_size,w*h,8))
    # pred_geo = tf.reshape(pred_geo,(batch_size,w*h,8))
    


    # pred_x_0 = gt_geo[:,:,0]
    # pred_x_2 = gt_geo[:,:,2]
    # pred_x_4 = gt_geo[:,:,4]
    # pred_x_6 = gt_geo[:,:,6]

    
    # result_0 = tf.maximum(gt_geo[:,:,0],pred_geo[:,:,0])
    # result_1 = tf.maximum(gt_geo[:,:,1],pred_geo[:,:,1])
    # result_4 = tf.minimum(gt_geo[:,:,4],pred_geo[:,:,4])
    # result_5 = tf.minimum(gt_geo[:,:,5],pred_geo[:,:,5])
    
    # result_2 = tf.minimum(gt_geo[:,:,2],pred_geo[:,:,2])
    # result_3 = tf.maximum(gt_geo[:,:,3],pred_geo[:,:,3])
    
    
    
    # result_6 = tf.maximum(gt_geo[:,:,6],pred_geo[:,:,6])
    # result_7 = tf.minimum(gt_geo[:,:,7],pred_geo[:,:,7])
    
    # iou = (result_5-result_1)*(result_4-result_0)
    # interface = tf.math.abs((pred_geo[:,:,5] - pred_geo[:,:,1])*(pred_geo[:,:,4] - pred_geo[:,:,0]))+tf.math.abs((gt_geo[:,:,5] - gt_geo[:,:,1])*(gt_geo[:,:,4] - gt_geo[:,:,0]))-tf.matg.abs(iou)
   
    # return -tf.math.log(tf.reshape(((iou+1)/(interface+1)),(batch_size,w,h,1))+ 1e-5 )
#     return tf.reduce_mean(gt_geo-pred_geo)


# In[38]:


#net  =  afterVGG16()#tf.keras.models.load_model('net.h5')#
net  =  tf.keras.models.load_model('net.h5')

# In[39]:


ops = keras.optimizers.Adam(2e-5,beta_1=0.5)


# In[40]:


@tf.function
def train_step(train_data,label_score_img,label_abgle_img,label_geo_img):
    #persistent true 为了重复计算梯度
    with tf.GradientTape(persistent=True) as tape:
        
        pred_score,angle_map,geo_map = net(train_data,training=True)
        
        loss_score = scoreMapLossDice(label_score_img,pred_score)
        loss_angle = lossAngle(label_abgle_img,angle_map)
        loss_geo = lossGeo(label_geo_img,geo_map)
        loss_all = loss_score + 10*loss_angle +loss_geo
        

    gradient = tape.gradient(loss_all,net.trainable_variables)
    
    ops.apply_gradients(zip(gradient,net.trainable_variables))
    
    print('4')
    return loss_score,loss_angle,loss_geo


# In[1]:


def fit(train_data,label_score_img,label_abgle_img,label_geo_img,epochs):
    i =0
    for epoch in range(epochs):
        print(epoch)
        for t,s,a,g in zip(train_data,label_score_img,label_abgle_img,label_geo_img):
            loss_score,loss_angle,loss_geo = train_step(t,s,a,g)
            if  loss_score == np.nan or  loss_angle==np.nan or loss_geo==np.nan or loss_score == np.inf or  loss_angle==np.inf or loss_geo==np.inf or loss_score == np.NINF or  loss_angle==np.NINF or loss_geo==np.NINF:
                    break
#             print('loss_geo.shape is {}'.format(loss_geo.size))
            print("i={},loss_score is {},loss_angle is {},loss_geo is {}".format(i,loss_score,loss_angle,loss_geo))
            i = i+1
            net.save('net.h5')
        if  loss_score == np.nan or  loss_angle==np.nan or loss_geo==np.nan or loss_score == np.inf or  loss_angle==np.inf or loss_geo==np.inf or loss_score == np.NINF or  loss_angle==np.NINF or loss_geo==np.NINF:
                                    break

# In[2]:


# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:

#     tf.config.experimental.set_memory_growth(gpu, True)
fit(train_data,label_score_data,label_angle_data,label_geo_data,500)


# In[ ]:


net.save('net.h5')


# In[ ]:





# In[ ]:




