import cv2
import numpy as np
 
# Read First Image
img1 = cv2.imread('GFG.png')
 
# Read Second Image
img2 = cv2.imread('GFG.png')
 
 
# concatenate image Horizontally
Hori = np.concatenate((img1, img2), axis=1)
 
# concatenate image Vertically
Verti = np.concatenate((Hori,img2), axis=0)
 
cv2.imshow('HORIZONTAL', Hori)
cv2.imshow('VERTICAL', Verti)
 
cv2.waitKey(0)
cv2.destroyAllWindows()