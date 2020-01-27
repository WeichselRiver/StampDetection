# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:59:40 2020

@author: cweic
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# load image
Filename = "ColorDetect1.png"
img = cv2.imread(Filename)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lowerBound=np.array([90,40,40])
upperBound=np.array([130,255,255])

mask=cv2.inRange(hsv,lowerBound,upperBound)


kernelOpen=np.ones((3,3))
kernelClose=np.ones((3,3))

maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

res = cv2.bitwise_and(img2,img2, mask= maskClose)

plt.imshow(res),plt.show()