import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def displayImg(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

def reMap(oldVal,oldMin,oldMax,newMin,newMax):
    oldRange = abs(oldMax-oldMin)
    newRange = abs(newMax-newMin)
    percentage = (oldVal-oldMin)/oldRange
    newVal = (newRange*percentage)+newMin
    return newVal

sao = pd.read_excel('C:/PythonPrograms/GitClones/CodingProgressforThesis/Star Tracker/DATA/SAO.xlsx')
sao_arr = sao.to_numpy() #Shape = (258997,4)

star_id = sao_arr[:,0].astype(int)
ra = sao_arr[:,1]
de = sao_arr[:,2]
ma = sao_arr[:,3]

image = np.zeros((720,1280))
ra_convert = np.round(reMap(ra,min(ra),max(ra),0,1280))
ra_convert = ra_convert.astype(int)
de_convert = np.round(reMap(de,min(de),max(de),0,720))
de_convert = de_convert.astype(int)

for i in star_id:
        cv2.circle(image,(ra_convert[i],de_convert[i]),1,(255,255,255),-1)

displayImg(image,cmap='gray')