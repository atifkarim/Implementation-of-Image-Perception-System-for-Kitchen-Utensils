import glob
import cv2
import numpy as np
import os

path_mask = '/media/atif/0820209220208930/unreal_cv_documentation/unreal_cv_image_processing_for_ML/object_subtraction_for_UE4/mask_image'
path_rgb = '/media/atif/0820209220208930/unreal_cv_documentation/unreal_cv_image_processing_for_ML/object_subtraction_for_UE4/RGB_image'

# img_path_mask = sorted(glob.glob(path_mask+ '/*.png'))
# img_path_rgb = sorted(glob.glob(path_rgb+ '/*.png'))
for a,b,image_mask in os.walk(path_mask):
    for s in image_mask:
        n=s.split(".")
        mask = path_mask+'/'+s
        image_mask2=cv2.imread(mask)
        
        hsv = cv2.cvtColor(image_mask2, cv2.COLOR_BGR2HSV)
        hsv_channels = cv2.split(hsv)

        _,thresh=cv2.threshold(hsv_channels[1],140,255,cv2.THRESH_BINARY_INV)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_mask2, contours, 0, (0,255,0), 3)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        
        for c,d,image_rgb in os.walk(path_rgb):
            for t in image_rgb:
                m=t.split('.')
                if int(n[0])== int(m[0]):
                    rgb = path_rgb+'/'+t
                    image_rgb2=cv2.imread(rgb)
                    image_rgb_rec=cv2.rectangle(image_rgb2,(x,y),(x+w,y+h),(255,255,255),1)
                    crop_img = image_rgb_rec[y:y+h, x:x+w]
                    cv2.imwrite("/media/atif/0820209220208930/unreal_cv_documentation/unreal_cv_image_processing_for_ML/object_subtraction_for_UE4/test_crop_RGB/"+str(t),crop_img)
                else:
                    pass
