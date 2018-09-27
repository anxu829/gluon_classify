# -*- coding:utf-8 -*-
import os , sys
import glob
import shutil
import cv2
import matplotlib.pyplot as plt
# import imgaug
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2
# from imgaug import augmenters as iaa


if os.path.exists('./TRAIN'):
    shutil.rmtree('./TRAIN')
os.mkdir('./TRAIN/')

if os.path.exists('./TEST'):
    shutil.rmtree('./TEST')
os.mkdir('./TEST/')

if os.path.exists('./VAL'):
    shutil.rmtree('./VAL')
os.mkdir('./VAL')


group = glob.glob('./datasets/*')
testpercent = 0.1
valpercent = 0.3
genNum = 4



def change_224_padding(img, size = 224):


    h, w, c = img.shape

    if h > w:
        h_new = size
        w_new = int(round(h_new * w / h) )
        img_ = cv2.resize(img, (w_new, h_new))  # 按比例缩放后的图片

        # 三通道分离，分别做pad变换
        # img_new储存了三个通道的六张灰度图片，其中索引为0、2、4的为一张图片，1、3、5的为另一张图片
        img_new = []
        for i in range(3):

            img_pad = img_[:, :, i]

            img_pad_ = np.pad(img_pad, h_new - w_new, 'reflect')
            img_pad1 = img_pad_[(h_new - w_new + 1): (h_new * 2 - w_new + 1), :h_new]
            img_pad2 = img_pad_[(h_new - w_new + 1): (h_new * 2 - w_new + 1), (h_new - w_new):]
            img_new.append(img_pad1)
            img_new.append(img_pad2)


        img_new1 = np.ones((size, size, 3))
        img_new2 = np.ones((size, size, 3))

        # img_new1和img_new2的通道顺序都是bgr
        img_new1[:, :, 0] = img_new[0]
        img_new1[:, :, 1] = img_new[2]
        img_new1[:, :, 2] = img_new[4]
        img_new2[:, :, 0] = img_new[1]
        img_new2[:, :, 1] = img_new[3]
        img_new2[:, :, 2] = img_new[5]


    elif h < w:
        w_new = size
        h_new = int(round(w_new * h / w))
        img_ = cv2.resize(img, (w_new, h_new))

        img_new = []
        for i in range(3):
            img_pad = img_[:, :, i]

            img_pad_ = np.pad(img_pad, w_new - h_new, 'reflect')
            img_pad1 = img_pad_[:w_new, (w_new - h_new + 1): (w_new * 2 - h_new + 1)]
            img_pad2 = img_pad_[(w_new - h_new):, (w_new - h_new + 1): (w_new * 2 - h_new + 1)]
            img_new.append(img_pad1)
            img_new.append(img_pad2)

        img_new1 = np.ones((size, size, 3))
        img_new2 = np.ones((size, size, 3))

        img_new1[:, :, 0] = img_new[0]
        img_new1[:, :, 1] = img_new[2]
        img_new1[:, :, 2] = img_new[4]

        img_new2[:, :, 0] = img_new[1]
        img_new2[:, :, 1] = img_new[3]
        img_new2[:, :, 2] = img_new[5]
    # 转换成整型
    else:
        img_new1 =  cv2.resize(img, (size, size))
        img_new2 =  cv2.resize(img, (size, size))
    img_new1 = np.array(img_new1, np.int32)
    img_new2 = np.array(img_new2, np.int32)

    return [img_new1, img_new2]


def change_size(img, r = 172, g = 172, b = 172, length = 224):

    # img = cv2.imread(path)
    h, w, c = img.shape

    if h >= w:
        h_new = length
        w_new = int(round(h_new * w / h))
        img_ = cv2.resize(img, (w_new, h_new))  # 按比例缩放后的图片

        # 背景
        back = np.ones((h_new, h_new - w_new, 3))
        back[:, :, 0] = back[:, :, 0] * r
        back[:, :, 1] = back[:, :, 1] * g
        back[:, :, 2] = back[:, :, 2] * b

        back1 = back[:, :int(round((h_new - w_new) / 2)), :]
        back2 = back[:, int(round((h_new - w_new) / 2)):, :]


        print(back1.shape, back2.shape, img_.shape)
        img_new = np.concatenate((back1, img_, back2), axis = 1)

    else:
        w_new = length
        h_new = int(round(w_new * h / w))
        img_ = cv2.resize(img, (w_new, h_new))  # 按比例缩放后的图片

        # 背景
        back = np.ones((w_new - h_new, w_new, 3))

        back[:, :, 0] = back[:, :, 0] * r
        back[:, :, 1] = back[:, :, 1] * g
        back[:, :, 2] = back[:, :, 2] * b


        back1 = back[:int(round((h_new - w_new) / 2)), :, :]
        back2 = back[int(round((h_new - w_new) / 2)):, :, :]


        #print(back1.shape, back2.shape, img_.shape)
        
        img_new = np.concatenate((back1, img_, back2), axis = 0)

    return img_new



for folder in group:
    global changeDict
    classes = glob.glob(folder + '/*')
    for class_ in classes:
        className = class_.split('/')[-1]
        print(className)
        fileList = glob.glob(class_ + '/*')
        testNum = int(testpercent*len(fileList))
        print(testNum)
        
        valNum = int(valpercent*len(fileList))
        print(valNum)
        tstSet = fileList[:testNum]
        valSet = fileList[testNum:valNum]
        trainSet = fileList[valNum:]

        if not os.path.exists('./TRAIN/'+className):
            os.mkdir('./TRAIN/'+className)
        if not os.path.exists('./VAL/'+className):
            os.mkdir('./VAL/'+className)
        if not os.path.exists('./TEST/'+className):
            os.mkdir('./TEST/'+className)

        # 两侧各自做5%的裁剪
        trimRate = 0.06
        for file in tstSet:
            fileName = file.split('/')[-1]
            img = cv2.imread(file)
            img = img[ int(img.shape[0] * trimRate) : int(img.shape[0] * (1-trimRate)) ,
                        int(img.shape[1] * trimRate) : int(img.shape[1] *(1- trimRate))
                    ]

            imgmean = img.mean()
            
            if max(img.shape[0] / img.shape[1] ,img.shape[1] / img.shape[0]  ) > 4:
                if img.shape[0] / img.shape[1] > img.shape[1] / img.shape[0] :
                    img = img[ int(img.shape[0] // 2  -  img.shape[0] * 0.125) :int( img.shape[0] // 2  +  img.shape[0] * 0.125),:]
                else:
                    img = img[:, int(img.shape[1] // 2  -  img.shape[1] * 0.125) :int( img.shape[1] // 2  +  img.shape[1] * 0.125)]

            # img = change_size( img  ,  r = imgmean, g = imgmean, b = imgmean, length = 224)
            # img = img.astype(int)
            imgaugList = change_224_padding(img)
            for idx,  imgaug in enumerate(imgaugList[:1]):
                # print(imgaug.shape)
                cv2.imwrite('./TEST/'+className + '/' + 'aug_' + str(idx) +  fileName,imgaug)
                
        for file in trainSet:            
            fileName = file.split('/')[-1]
            img = cv2.imread(file)
            img = img[ int(img.shape[0] * trimRate) : int(img.shape[0] * (1-trimRate)) ,
                        int(img.shape[1] * trimRate) : int(img.shape[1] *(1- trimRate))
                    ]
            if img.shape[0] * img.shape[1] > 4000 :
                imgmean = img.mean()
                
                if max(img.shape[0] / img.shape[1] ,img.shape[1] / img.shape[0]  ) > 4:
                    if img.shape[0] / img.shape[1] > img.shape[1] / img.shape[0] :
                        img = img[ int(img.shape[0] // 2  -  img.shape[0] * 0.125) :int( img.shape[0] // 2  +  img.shape[0] * 0.125),:]
                    else:
                        img = img[:, int(img.shape[1] // 2  -  img.shape[1] * 0.125) :int( img.shape[1] // 2  +  img.shape[1] * 0.125)]

                # img = change_size( img  ,  r = imgmean, g = imgmean, b = imgmean, length = 224)
                # img = img.astype(int)
                imgaugList = change_224_padding(img)
                for idx,  imgaug in enumerate(imgaugList[:1]):
                    # print(imgaug.shape)
                    cv2.imwrite('./TRAIN/'+className + '/' + 'aug_' + str(idx) +  fileName,imgaug)
            
        for file in valSet:            
            fileName = file.split('/')[-1]
            img = cv2.imread(file)
            img = img[ int(img.shape[0] * trimRate) : int(img.shape[0] * (1-trimRate)) ,
                        int(img.shape[1] * trimRate) : int(img.shape[1] *(1- trimRate))
                    ]
            imgmean = img.mean()
            if max(img.shape[0] / img.shape[1] ,img.shape[1] / img.shape[0]  ) > 4:
                if img.shape[0] / img.shape[1] > img.shape[1] / img.shape[0] :
                    img = img[ int(img.shape[0] // 2  -  img.shape[0] * 0.125) :int( img.shape[0] // 2  +  img.shape[0] * 0.125),:]
                else:
                    img = img[:, int(img.shape[1] // 2  -  img.shape[1] * 0.125) :int( img.shape[1] // 2  +  img.shape[1] * 0.125)]

            # img = change_size( img  ,  r = imgmean, g = imgmean, b = imgmean, length = 224)
            # img = img.astype(int)
            imgaugList = change_224_padding(img)
            for idx,  imgaug in enumerate(imgaugList[:1]):
                # print(imgaug.shape)
                cv2.imwrite('./VAL/'+className + '/' + 'aug_' + str(idx) +  fileName,imgaug)
            


