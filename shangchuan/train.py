#!/usr/bin/env python
# coding: utf-8

# In[1]:



import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import  matplotlib.pyplot as plt 

import glob
import pandas as pd
import cv2
import os
#import logging
#logging.disable(30)


# In[2]:


tf.__version__


# In[3]:


# 占位符号 占的是Python数据集
#(images ,labels),(_,_) = tf.keras.datasets.mnist.load_data()

image_path= glob.glob('new_*/*.jpg')
len(image_path)


# In[4]:


image_path[-5:]


# In[5]:


#乱序
np.random.seed(20)
np.random.shuffle(image_path)


# In[6]:


image_path[-5:]


# In[7]:


#将路径中的man or woman 提取出来 作为标签
labels = [p.split('_')[1].split('/')[0] for p in image_path]


# In[8]:


np.unique(labels)# 取出一共有多少分类


# In[9]:


#将label转化为序号
class_to_num = dict((name ,i) for i,name in enumerate(np.unique(labels)) )#dict 将其内部转化为序号形式
num_to_class = dict((name,i) for i,name in class_to_num.items())#{0:'man',1:'woman'} ,映射回man or woman


# In[10]:


labels = [class_to_num.get(name) for name in labels]


# In[11]:


labels[-5:]


# In[12]:


class_num = len(np.unique(labels))


# In[13]:


class_num


# In[14]:


labels


# In[15]:


#t = []
#for i in labels:
 #       subLabels =[0,0,0,0,0,0,0,0,0,0]
  #      subLabels[i] = 1.0
   #     t.append(subLabels)
#labels = t     
labels = pd.get_dummies(labels,sparse=True)


# In[16]:


len(labels)


# In[17]:


labels


# In[18]:


#转化为array
image_path = np.array(image_path)
labels = np.array(labels)


# In[ ]:





# In[19]:


def load_images(path):
    imgs = tf.io.read_file(path)
    imgs = tf.image.decode_jpeg(imgs,channels=3)
    imgs = tf.image.resize(imgs,[80,80])# 产生畸变
#     imgs = tf.image.random_crop(imgs,[64,64,3])
    #imgs = tf.image.random_flip_left_right(imgs) #左右翻转
    imgs = imgs/127.5-1
    return imgs


# In[20]:


images_dataset = tf.data.Dataset.from_tensor_slices(image_path)#创建image的dataset
images_dataset = images_dataset.map(load_images)


# In[21]:


images_dataset


# In[22]:


label_dataset = tf.data.Dataset.from_tensor_slices(labels)#创建image的dataset


# In[23]:


label_dataset


# In[24]:


dataset = tf.data.Dataset.zip((images_dataset,label_dataset))


# In[25]:


dataset


# In[26]:


batch_size = 50
noise_dim = 50
image_count = len(image_path)


# In[27]:


dataset = dataset.shuffle(image_count).batch(batch_size)


# In[28]:


dataset


# In[ ]:





# In[29]:


def discriminator():
    image = layers.Input(shape=(80,80,3))
    print('21')
    # 接下里进行卷积el    # 辨别器使用dropout  可以让进入的数据不那么多，从而是辨别器和生成其器进行抵抗                     
    x = layers.Conv2D(64,(3,3),strides=(2,2),padding='same',use_bias= False)(image)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)
    print('22')
    x = layers.Conv2D(64*2,(3,3),strides=(2,2),padding='same',use_bias= False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)
    print('23')                     
    x = layers.Conv2D(64*4,(3,3),strides=(2,2),padding='same',use_bias= False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    print('24') 
    x = layers.Conv2D(64*8,(3,3),strides=(2,2),padding='same',use_bias= False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    print('25')                      
    x = layers.Flatten()(x) 
    #x1 = layers.Dense(1)(x)#   真假输出
    x2 = layers.Dense(11)(x) # 分类输出 一共与10类
    print('26') 
    model = keras.Model(inputs=image,outputs = [x2])#,x2])
                         #logits 表示为激活的结果
    return model


# In[30]:


disc = discriminator()


# In[31]:


bce = keras.losses.BinaryCrossentropy(from_logits  = True)
cce = keras.losses.CategoricalCrossentropy(from_logits = True)#对label进行0-1编码的，没有进行one-hot的


# In[32]:


#def get_loss(inputs_real,condition_label,smooth=0.1):
 #   d_logits_real,d_outputs_real = discriminator(inputs_real)
    #计算loss
  #  d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real,labels = tf.ones_like(d_outputs_real)*condition_label))
   # return d_loss
def disc_loss(real_class_out,label):
    print('real_class_out={}'.format(real_class_out))
    print('label={}'.format(label))
    real_loss = cce(label,real_class_out)
    print(real_loss)
    #class_loss = cce(label,real_class_out) # y_true, y_pred 分类损失
    return real_loss  
    
    


# In[33]:


disc_opt = keras.optimizers.Adam(1e-5)


# In[34]:


#def get_optimizer(d_loss,beta =0.4,learning_rate=0.001):
 #   train_vars = tf.trainable_variables()
  #  d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

   # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #    d_opt = tf.train.AdamOptimizer(learning_rate,beta1=beta).minimize(d_loss,var_list = d_vars)
    #return d_opt
# 训练函数，接受一个批次的数据
@tf.function 
def train_step(image,label):
    #正态分布的随机
    #为了解决总体数据/batch_size不可以除尽的问题,使用和label相同大小的数据
    size = label.shape[0]
    print('0')
    noise = tf.random.normal([size,noise_dim])
    # 使用梯度,建立上下文管理器，一旦创建上下文管理器，就会记录梯度的计算过程，当需要取计算
    #梯度的时候，会自动按照次计算过程进行计算
    #tape是记录计算过程
    print('1')
    with tf.GradientTape() as disc_tape:
        print('2')
        print('4')
        real_class_out= disc(image,training=True)
        print('5')
        #label = tf.image.random_crop(label,[batch_size,10])
        print(label)
        disc_loss_ = disc_loss(real_class_out,label)
        
        print('disc_loss_={}'.format(disc_loss_))
        print('6')
        
        print('7')
        
        # 计算生成器损失值和 生成器参数之间的梯度，计算判别器损失值和判别器参数之间的梯度
    print(disc.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss_,disc.trainable_variables)
    print('8')
    
    #根据计算得到的梯度 优化变量
    
    disc_opt.apply_gradients(zip(disc_grad,disc.trainable_variables))
    print('9')
        


# In[35]:


def train(dataset,epoches):
    for epoch in range(epoches):
        if epoch %5 ==0:
            disc.save('tenClass_2.h5')
        print('11')
        for image_batch,label_batch in dataset:
            print('22')
            train_step(image_batch,label_batch)
        #if epoch%10 == 0:    
            #plot_gen_image(gen,noise_seed,label_seed,epoch)
    #plot_gen_image(gen,noise_seed,label_seed,epoches) 


# In[36]:


for image_batch,label_batch in dataset:
            print('22')
            real_class_out= disc(image_batch,training=True)
            print(label_batch)
            print(real_class_out)
            break


# In[37]:


real_class_out


# In[38]:


train(dataset,500)


# In[39]:


disc.save('tenClass_2.h5')


# 

# In[29]:


disc = discriminator()#tf.keras.models.load_model('tenClass_1.h5')


# In[30]:


for image_batch,label_batch in dataset:
            print('22')
            real_class_out= disc(image_batch,training=False)
            print(real_class_out)
            print(label_batch)
            break


# In[31]:


label_n = [0,0,0,0,0,0,0,0,0,0]
def split_resize_img(img,label):
    # 去掉边框
    print(img.shape)
    # 0:y 1:x
    #image = image[40:110, 35:190]
    image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 按照所占比重数量进行切割
    # 100以上为阈值 100以下
    counter = 0
    begin_y = 0
    end_y = 0
    begin_x=0
    end_x=0
    x = image.shape[0]
    y = image.shape[1]
    #print(x)
    #print(y)
    #行不变,列从小到大
    for i in range(x):
            for j in range(y):
                    #print(image[i,j])
                    if image[i,j] >100:
                            counter+=1
            #print('counter1={}'.format(counter))                
            if(y>2*counter):
                    begin_x = i
                    i = y+1
                    break
            counter = 0
    counter = 0  
    #航不变，列从大到小 
    i=0
    j=0             
    for i in range(x):
            for j in range(y):
                    #print(image[x-i-1,j])
                    if image[x-i-1,j] >100:
                            counter+=1
            #print('counter2={}'.format(counter))                
            if(y>2*counter):
                    end_x = x-i-1
                    break                
            counter = 0
    counter = 0
    #列不变 航从大到小
    for j in range(y):
            for i in range(x):
                    if image[i,j] >100:
                            counter+=1
            #print('counter3={}'.format(counter)) 
            if(x>2*counter):
                    begin_y = j
                    j = y+1
                    break
            counter = 0        
    counter = 0
    for j in range(y):
            for i in range(x):
                    if image[i,y-j-1] >100:
                            counter+=1
            #print('counter4={}'.format(counter)) 
            if(x>2*counter):
                    end_y = y-j-1
                    j = y+1
                    break                
            counter = 0 
    print(begin_x)
    print(end_x)         
    print(begin_y)
    print(end_y)
    
    cv2.imshow("img_gray1",image)
    #image = image[begin_x:end_x, begin_y:end_y]
    image = image[40:110, 35:190]
    _,image = cv2.threshold(image,170,255,0)
    begin_y = find_begin_y(image,5)
    #print('begin_y={}'.format(begin_y))
    end_y = find_end_y(image,5)
    #print('end_y={}'.format(end_y))
    image = image[0:image.shape[0], begin_y:end_y]
    label = label.split('/')[-1].split('_')[0]
    print('label={}'.format(label))
    label_size = len(label)
    subLen = int(image.shape[1]/label_size)
    #print(end_y-begin_y)
    #print(subLen)
    cv2.imshow("img_gray",image)
    img1 = image[0:image.shape[0], 0:int(subLen)]
    
    
    cv2.imshow("img",img1)
    t  = (label_n[int(label[0])])
    if os.path.exists('new_'+str(label[0])) == False:
                os.makedirs('new_'+str(label[0]))
    cv2.imwrite('new_'+str(label[0])+'/'+str(t)+'.jpg', img1)
    label_n[int(label[0])]+=1
    print('label[0]={}'.format(label[0]))

    if label_size >=2 :
        print('label[1]={}'.format(label[1]))    
        img2 = image[0:image.shape[0], int(subLen):int(subLen*2)]
        cv2.imshow("img2",img2)
        t  = (label_n[int(label[1])])
        if os.path.exists('new_'+str(label[1])) == False:
                os.makedirs('new_'+str(label[1]))
        cv2.imwrite('new_'+str(label[1])+'/'+str(t)+'.jpg', img2)
        label_n[int(label[1])]+=1


    if label_size==3 :
        print('label[2]={}'.format(label[2]))    
        img3 = image[0:image.shape[0], int(subLen*2):int(subLen*3)]
        cv2.imshow("img3",img3)
        t  = (label_n[int(label[2])])
        if os.path.exists('new_'+str(label[2])) == False:
                os.makedirs('new_'+str(label[2]))
        cv2.imwrite('new_'+str(label[2])+'/'+str(t)+'.jpg', img3)
        label_n[int(label[2])]+=1
    cv2.waitKey(120)
    return image


# In[32]:


def find_begin_y(image,size=1):
    begin_y = 0   
    counter = 0 
    x = image.shape[0]
    y = image.shape[1]
    #print(x)
    #print(y)
    #lie不变,hang从小到大
    for j in range(y):
            for i in range(x):
                    #print(image[i,j])
                    if image[i,j] ==255:
                            counter+=1
            #print('counter1={}'.format(counter))                
            if(counter>size):
                    begin_y = j
                    i = y+1
                    break
    return begin_y


# In[33]:


def find_end_y(image,size=1):
    end_y = 0   
    counter = 0 
    x = image.shape[0]
    y = image.shape[1]
    #print(x)
    #print(y)
    j = y-1
    #lie不变,hang从小到大
    while j>=0:
            for i in range(x):
                    #print(image[i,j])
                    if image[i,j] ==255:
                            counter+=1
            #print('counter1={}'.format(counter))                
            if(counter>size):
                    end_y = j
                    break
            j-=1        
    return end_y


# In[34]:


def split_img(path,name):
    result =[]
    img = cv2.imread(path)
    #cv2.imshow("img_ori",img)
    img = cv2.resize(img, (0, 0), fx=300/img.shape[1], fy=300/img.shape[0], interpolation=cv2.INTER_NEAREST)
    print(img.shape)
    #for i in range(img.shape[0]):
    #   for j in range(img.shape[1]):
            
    #      if(img[i][j][0]+img[i][j][1]+img[i][j][2])/3 < 200:
    #         img[i][j] =0 
    #cv2.imshow("first",img)            
    #img = img[50:300, 100:300]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #edge = cv2.Canny(img, threshold1,threshold2[, edges[, apertureSize[, L2gradient ]]])
    #lines = cv2.HoughLinesP(edges, 1,np.pi/180, 80, minLineLength, maxLineGap)

    #img1 = np.float32(img)
    #img1 = cv2.cornerHarris(img1, 2, 3, 0.01)
    #img1[img1 > 0.01*img1.max()] = (0, 0, 255)

    #sum/=(img.shape[0]*img.shape[1])
    #print(sum)        

    _,img = cv2.threshold(img,170,255,0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, kernel)
    img = cv2.erode(img, kernel)
    cv2.imshow("img_gray",img)

    contours , hierarchy = cv2.findContours ( img , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    img2 =img
    # for cnt in contours:
    #      x, y, w, h = cv2.boundingRect(cnt) 
    #      cv2.rectangle(img2, (x,y), (x+w,y+h), (170,233,30), 2)
    # cv2.imshow("img_gray2",img2)
    print(len(contours))


    print("contours.size()=".format(len(contours)))
    i = 0
    big_cnt = contours[0]
    x_big, y_big, w_big, h_big = cv2.boundingRect(big_cnt)
    td = 0
    i=0
    list_contours = []

    contours.sort(key=takeArea,reverse=True)
    counter_contian = 0
    list_counter_contian =[]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        for cnt1 in contours:
            x1,y1,w1,h1 = cv2.boundingRect(cnt1)
            if w1*h1<100 or w1<10 or h1<10:
                list_counter_contian.append(cnt1)
                continue
            if x == x1 and y ==y1 and w==w1 and h==h1:
                continue
            
            
            if x1 >x and x1<x+w or x1+w1 >x and x1+w1<x+w :
                if y1 >y and y1<y+h or y1+h1 >y and y1+h1<y+h:
                    print(x,y,w,h)
                    print(x1,y1,w1,h1)
                    print('\n')
                    list_counter_contian.append(cnt)
                    break
            #if w*h<100:
             #   if all(cnt in list_counter_contian):
              #      continue
               # list_counter_contian.append(cnt)      
        
    print("面积")    
    print(len(list_counter_contian))

    for cnt in list_counter_contian:
        x, y, w, h = cv2.boundingRect(cnt)
        print(x,y,w,h)    
    print("面积")
    
    
    contours.sort(key=takeSecond,reverse=False)
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     if x*y <100 or x <10 or y<10:
    #         list_contours.append(i)
    #         i+=1
    #         continue   
            
    #     #print(x,x+w,y,y+h)
    #     if w >= w_big :
    #         if  h >= h_big:
    #             #if x<x_big:
    #                 #if y<y_big:
    #                     big_cnt = cnt
    #                     x_big, y_big, w_big, h_big = cv2.boundingRect(cnt) 
    #                     list_contours.append(i)
    #                     td = i
    #     i+=1                
    #img1 = img[x_big:x_big+h_big, y_big:y_big+w_big] 
    #contours , hierarchy = cv2.findContours ( img1 , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    #img  = img1
    #cv2.imwrite('my/'+'hehe.jpg', img1)

    print("list_counter_contian.size()={}".format(len(list_counter_contian)))
    print("name = {}".format(name))
    
    t = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) 
        if w*h<100 or w<10 or h<10:
            continue    
        # print(type(cnt in list_counter_contian))    
        if listInLists(cnt,list_counter_contian) :
            t+=1
            continue
        else:    
            print(x,y,w,h)   
            #cv2.rectangle(img, (x,y), (x+w,y+h), (170,233,30), 2)
            img1 = img[ y:y+h,x:x+w]
            #img1 =(img1-127.5)/127.5
            result.append('my/'+str(path.split('/')[2].split('.')[0])+'_'+str(name)+'.jpg')
            #if os.path.exists(path.split('/')[1].split('.')[0]) == False:
              #  os.makedirs(path.split('/')[1].split('.')[0])
            print(str(path.split('/')))
            cv2.imwrite('my/'+str(path.split('/')[2].split('.')[0])+'_'+str(name)+'.jpg', img1)#[x:x+h, y:y+w])
            cv2.waitKey(5)
            #if name == 28:
             #   exit()
            name+=1    
        t+=1
        
    cv2.imshow("Contours",img)#img[x_big:x_big+w_big, y_big:y_big+h_big])

    # cv2.drawContours(img,contours,-1,(176,0,255),3) 
    # cv2.imshow("Contours",img)
    cv2.waitKey(500)
    return name,result


# In[48]:


#image_path= glob.glob('./jpg/*.jpg')

#img = cv2.imread('2123824676.jpg')
#split_resize_img(img,'123_.jpg')


# In[ ]:




