# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 07:37:58 2020

@author: cweic
"""

from __future__ import print_function
import cv2
import numpy as np
 
 
MAX_FEATURES = 40000
GOOD_MATCH = 100
 
 
def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(nfeatures = MAX_FEATURES, scaleFactor = 1.2,
                       nlevels = 4, edgeThreshold = 20,
                       firstLevel = 0, WTA_K = 3,
                       patchSize = 20, fastThreshold = 20)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:GOOD_MATCH]
  SumMatches = 0
  for m in matches:
      #print(m.distance)
      SumMatches = SumMatches+m.distance
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  M, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
 
  # Use homography
  h,w = im1Gray.shape
  pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
  dst = cv2.perspectiveTransform(pts,M)
  im1Reg = cv2.polylines(im2Gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
  #im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, SumMatches
 
 
if __name__ == '__main__':
   
  # Read reference image
  refFilename = "Find2.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
 
  # Read image to be aligned
  imFilename = "Orig4.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, SumMatches = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
 
  # Print estimated homography
  print("Estimated Quality : \n",  SumMatches)