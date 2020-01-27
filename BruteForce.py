# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 19:13:00 2020

@author: cweic
"""
import cv2
import numpy as np

img1 = cv2.imread("Find.jpg", cv2.IMREAD_GRAYSCALE).astype(np.uint8)
img2 = cv2.imread("Orig.jpg", cv2.IMREAD_GRAYSCALE).astype(np.uint8)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
ImgKp1 = cv2.drawKeypoints(img1, kp1, None)
ImgKp2 = cv2.drawKeypoints(img2, kp2, None)



cv2.imshow("Image1", ImgKp1)
cv2.imshow("Image2", ImgKp2)
#cv2.imshow("Image2", img2)
cv2.imshow("Matching Result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
