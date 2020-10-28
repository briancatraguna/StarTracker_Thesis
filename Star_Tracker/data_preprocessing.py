#Importing the necessary libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import imutils

def displayImg(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

#Putting the image into a list
images = []
for index in range(8):
    path = 'C:/PythonPrograms/GitClones/CodingProgressforThesis/Star Tracker/dataset/'+str(index)+'/'
    img = cv2.imread(path+'0.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    images.append(img)
    displayImg(img,cmap='gray')

#Rotation variance
for index,image in enumerate(images):
    count=0
    path = 'C:/PythonPrograms/GitClones/CodingProgressforThesis/Star Tracker/dataset/'+str(index)+'/'
    for angle in np.arange(0,360,3):
        count+=1
        rotated = imutils.rotate_bound(image,angle)
        cv2.imwrite(path+str(count)+'.jpg',rotated)

path = 'C:/PythonPrograms/GitClones/CodingProgressforThesis/Star Tracker/dataset/'
#Noise variance
for folder in os.listdir(path):
    count=120
    for filename in os.listdir(path+folder+'/'):
        img = cv2.imread(path+folder+'/'+filename)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        row,col = img_gray.shape
        for max_noise_step in range(1,4):
            max_noise = 50*max_noise_step
            noise = np.random.randint(0,max_noise,size=(row,col)).astype('uint8')
            for addweighted_step in range(0,5):
                count+=1
                minus_weighted = addweighted_step*0.1
                dst = cv2.addWeighted(img_gray,0.9-minus_weighted,noise,0.1+minus_weighted,0)
                cv2.imwrite(path+folder+'/'+str(count)+'.jpg',dst)

#Threshold variation
for folder in os.listdir(path):
    count=1935
    for i in range(121):
        img = cv2.imread(path+folder+'/'+str(i)+'.jpg')
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for i in range(5):
            thresh_value = i*20
            ret1, th1 = cv2.threshold(gray_img,100+thresh_value,255,cv2.THRESH_BINARY)
            count+=1
            cv2.imwrite(path+folder+'/'+str(count)+'.jpg',th1)
            ret2, th2 = cv2.threshold(gray_img,100+thresh_value,255,cv2.THRESH_OTSU)
            count+=1
            cv2.imwrite(path+folder+'/'+str(count)+'.jpg',th2)

#Filter variation (Blur and sharpen)
for folder in os.listdir(path):
    count=3145
    for i in range(121):
        img = cv2.imread(path+folder+'/'+str(i)+'.jpg')
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for i in range(2,10,2):
            ksize=(i,i)
            blurredimg = cv2.blur(gray_img,ksize)
            count+=1
            cv2.imwrite(path+folder+'/'+str(count)+'.jpg',blurredimg)
        for i in range(9,12,1):
            kernel = np.array([[-1,-1,-1], [-1,i,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(gray_img,-1,kernel)
            count+=1
            cv2.imwrite(path+folder+'/'+str(count)+'.jpg',sharpen)