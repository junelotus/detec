import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np
import cv2
# import map

def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(50)
# w = int(1280/4)
# h = int(720/4)

img_ori = glob.glob('./1_ch4_training_images/*.jpg')
label_ori = glob.glob('./2_ch4_training_localization_transcription_gt/*.txt')
img_ori.sort()
label_ori.sort()
def swap(t1, t2):
    return t2, t1
def f(x):
    return int(x)
def drawPictureUseCornerDotOri(score_map,angle_map,geo_map,canvas_point_float):
    canvas_point  =[]# map(f,canvas_point_float)#int(canvas_point_float)
    # print(canvas_point_float)
    for elem in canvas_point_float:
        canvas_point.append((int(elem[0]),int(elem[1])))
    # print('canvas_point',canvas_point)

    if canvas_point[3][0] == canvas_point[0][0]:
        u_k = 0
    else :
        u_k = (canvas_point[3][1]-canvas_point[0][1])/(canvas_point[3][0]-canvas_point[0][0])
    u_d = canvas_point[3][1] - u_k*canvas_point[3][0]
    #print("u",u_k,u_d)

    if canvas_point[2][0]==canvas_point[1][0]:
        b_k = 0
    else:
        b_k = (canvas_point[2][1]-canvas_point[1][1])/(canvas_point[2][0]-canvas_point[1][0])
    b_d = canvas_point[2][1] - b_k*canvas_point[2][0]
    #print("b",b_k,b_d)

    if canvas_point[1][0] == canvas_point[0][0]:
        l_k = 0
        l_d = canvas_point[0][0]
    else :
        l_k = (canvas_point[1][1]-canvas_point[0][1])/(canvas_point[1][0]-canvas_point[0][0])
        l_d = canvas_point[0][1] - l_k*canvas_point[0][0]
    #print("l",l_k,l_d)

    if canvas_point[2][0] == canvas_point[3][0]:
        r_k = 0
        r_d = canvas_point[2][0]
    else:
        r_k =  (canvas_point[2][1]-canvas_point[3][1])/(canvas_point[2][0]-canvas_point[3][0])
        r_d = canvas_point[2][1] - r_k*canvas_point[2][0]
    #print("r",r_k,r_d)

    # x = 767
    # y = 47
    #print('bool',y-u_k*x-u_d >= 0  , y-b_k*x-b_d <= 0  , ((l_k > 0 and y - l_k*x-l_d <0 )or(l_k < 0 and y-l_k*x-l_d > 0 ) or(l_k==0 and x>=l_d)) , ((r_k>0 and y-r_k*x-r_d >0)or(r_k < 0 and y-r_k*x-r_d < 0) or(r_k==0 and x<=r_d) ))
    # x= 0
    # y =0
    for x in range(score_map.shape[1]):
        for y in range(score_map.shape[0]):
            # #print('bool',y-u_k*x-u_d >= 0  , y-b_k*x-b_d <= 0  , ((l_k > 0 and y - l_k*x-l_d <0 )or(l_k < 0 and y-l_k*x-l_d > 0 ) or(l_k==0 and x>=l_d)) , ((r_k>0 and y-r_k*x-r_d >0)or(r_k < 0 and y-r_k*x-r_d < 0) or(r_k==0 and x<=r_d) ))
            if  y-u_k*x-u_d >= 0  and y-b_k*x-b_d <= 0  and ((l_k==0 and x>=l_d) or (l_k > 0 and y - l_k*x-l_d <0 )or(l_k < 0 and y-l_k*x-l_d > 0 ) ) and ((r_k==0 and x<=r_d) or (r_k>0 and y-r_k*x-r_d >0)or(r_k < 0 and y-r_k*x-r_d < 0)  ):
                # print('0000000')
                score_map[y][x] = 255
                geo_map[0][y][x] = canvas_point_float[0][0]*0.4
                geo_map[1][y][x] = canvas_point_float[0][1]*0.4
                geo_map[2][y][x] = canvas_point_float[1][0]*0.4
                geo_map[3][y][x] = canvas_point_float[1][1]*0.4
                geo_map[4][y][x] = canvas_point_float[2][0]*0.4
                geo_map[5][y][x] = canvas_point_float[2][1]*0.4
                geo_map[6][y][x] = canvas_point_float[3][0]*0.4
                geo_map[7][y][x] = canvas_point_float[3][1]*0.4
                # print(geo_map[0][y][x])
                if b_k == 0:
                    angle_map[y][x] = 0
                elif b_k < 0 :
                    angle_map[y][x] = 90-abs(180*math.atan(b_k))
                else:
                    angle_map[y][x] = 180*math.atan(b_k)

    # showImg('score_map',score_map)
    # cv2.waitKey(1000)
    # cv2.imencode('.jpg', score_map)[1].tofile('score_map.jpg')
    # print('geo_map[0]',geo_map[0])
    return score_map,angle_map,geo_map
def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(1000)

for path_img in img_ori:

    index = path_img.split('/')[2].split('.')[0].split('_')[-1]
    print(index)
    # if (int(index) == 1000 ) or (int(index) >= 100 and int(index)/100 < 7  ) or (int(index) < 100 and int(index)/10 < 7) or( int(index) < 7):
    #     continue
    print(path_img)
    score_map = np.zeros((int(720/4),int(1280/4),1),dtype=np.uint8)#img_to_correct[0:height,0:width]
    angle_map = np.zeros((int(720/4),int(1280/4),1),dtype=np.float32)
    geo_map = []
    for i in range(8):

        geo_map.append(np.zeros((int(720/4),int(1280/4),1),dtype=np.float32))
    # print('len',len(geo_map))

    file = open('./2_ch4_training_localization_transcription_gt/gt_img_'+str(index)+'.txt',encoding='UTF-8-sig')
    img_to_correct = cv2.imread(path_img)
    while True:
        line = file.readline()
        if not line:
            break
        line = np.array(line.split(',')[0:8])
        line = line.astype(np.int32)
        line = line.reshape(-1,2)
        line = (line/4.0).astype(np.float32)

        #to construct the score_map angle_map geo_map,the result is three vector
        # angle_gt_sub  = np.zeros((w,h,1),dtype=np.float32)
        # findAngle(line,angle_gt_sub)
        t= line[1].copy()
        line[1] = line[3].copy()
        line[3] = t
        # line[3],line[1] = swap(,line[3])
        #print(line)
        score_map,angle_map,geo_map = drawPictureUseCornerDotOri(score_map,angle_map,geo_map,line)
    # print(path_img)
    # showImg('score map',score_map)
    # showImg('angle_map',angle_map)
    # cv2.imencode('.jpg', score_map)[1].tofile('score_map_1/'+str(index)+'.jpg')
    # cv2.imencode('.jpg', score_map)[1].tofile('angle_map_1/'+str(index)+'.jpg')
    for i in range(8):
        cv2.imencode('.jpg', geo_map[i])[1].tofile('geo_map_1/'+str(index)+'_'+str(i)+'.jpg')
    file.close()

img_all = glob.glob('./angle_map/*.jpg')
# for i in range(1000):
#     if i == 0:
#          continue
#     print('./angle_map/'+str(i)+'.jpg')
#     img  = cv2.imread('./angle_map/'+str(i)+'.jpg')
#     showImg(str(i),img)


