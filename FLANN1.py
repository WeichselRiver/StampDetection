# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:51:14 2020

@author: cweic
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 100
MAX_FEATURES = 20000

img1 = cv2.imread('Orig4.jpg', 0)  # queryImage
img2 = cv2.imread('Find2.jpg', 0) # trainImage

#img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
# 
orb = cv2.ORB_create(nfeatures = MAX_FEATURES, scaleFactor = 1.2,
                       nlevels = 4, edgeThreshold = 20,
                       firstLevel = 0, WTA_K = 3,
                       patchSize = 20, fastThreshold = 20)



kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

des1 = np.float32(des1)
des2 = np.float32(des2)

# matches = flann.knnMatch(des1, des2, 2)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>3:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

    if M is None:
        print ("No Homography")
    else:
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
