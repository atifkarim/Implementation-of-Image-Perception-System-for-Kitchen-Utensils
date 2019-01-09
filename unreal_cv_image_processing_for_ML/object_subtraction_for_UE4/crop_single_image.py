import numpy as np
import cv2

rgb=cv2.imread('/media/atif/0820209220208930/unreal_cv_documentation/unreal_cv_image_processing_for_ML/object_subtraction_for_UE4/214_rgb.png')
im = cv2.imread('/media/atif/0820209220208930/unreal_cv_documentation/unreal_cv_image_processing_for_ML/object_subtraction_for_UE4/214_mask.png')
# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) #convert to hsv

hsv_channels = cv2.split(hsv) #split the hsv color

rows = im.shape[0]
cols = im.shape[1]

# cv2.imshow("hsv_channels[1]",hsv_channels[1])  #displaying hsv image. also work for hsv_channels[0]

_,thresh=cv2.threshold(hsv_channels[1],140,255,cv2.THRESH_BINARY_INV) #do thresholding
# cv2.imshow('thresholded_image',thresh) #display thresholded image

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #apply contour function on thresholded image
# cnt = contours[4]
#You will find contour from thresholded or canny edged image clearly. After then you can apply the detected contour on other image. Ths has done later.

cv2.drawContours(im, contours, 0, (0,255,0), 3) #draw contour on masked image
# cv2.imshow('contoured_image_on_mask',im) #displaying contour on masked image

cv2.drawContours(hsv_channels[1], contours, 0, (0,255,0), 3) #draw contour on hsv image
# cv2.imshow('contoured_image_hsv',hsv_channels[1]) # Displaying contour on hsv image

cnt = contours[0]

# M = cv2.moments(cnt)
# print( M )

# k = cv2.isContourConvex(cnt)
# print(k)

x,y,w,h = cv2.boundingRect(cnt) #this related to bound the contour on a box. It actually takes information from contour

# cv2.rectangle(hsv_channels[1],(x,y),(x+w,y+h),(0,255,0),2) #draw this box on hsv image.
# cv2.imshow('rec_hsv[1]',hsv_channels[1])

# cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2) #draw this on thresholded image. result not good
# cv2.imshow('rec_thresh',thresh)

mask_rec=cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2) #draw this box on masked image
cv2.imshow('rec_mask',mask_rec)

rgb_rec=cv2.rectangle(rgb,(x,y),(x+w,y+h),(0,255,0),2) #draw this box on rgb image
cv2.imshow('rec_rgb',rgb_rec)

crop_img = rgb_rec[y:y+h, x:x+w] #to crop the applied bounded box
cv2.imshow("cropped", crop_img) #displaying cropped image
cv2.imwrite("/media/atif/0820209220208930/unreal_cv_documentation/unreal_cv_image_processing_for_ML/object_subtraction_for_UE4/crop_rgb.png",crop_img) #save the cropped image

if cv2.waitKey() == ord('q'): #press q to close the output image window
    cv2.destroyAllWindows()
