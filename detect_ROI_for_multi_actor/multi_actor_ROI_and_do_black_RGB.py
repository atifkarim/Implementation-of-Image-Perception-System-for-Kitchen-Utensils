import numpy as np
import cv2
import os

lit = 'F:/unreal_cv_documentation/detect_ROI_for_multi_actor/image_test/lit_1_1.png'
mask = 'F:/unreal_cv_documentation/detect_ROI_for_multi_actor/image_test/mask_1.png'


img = cv2.imread(mask)
rgb_img = cv2.imread(lit)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## Gen lower mask (0-5) and upper mask (175-180) of RED
mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

## Merge the mask and crop the red regions
mask = cv2.bitwise_or(mask1, mask2 )
croped = cv2.bitwise_and(img, img, mask=mask)
f = open('F:/unreal_cv_documentation/detect_ROI_for_multi_actor/image_test/ROI.txt', 'a')
f = open('F:/unreal_cv_documentation/detect_ROI_for_multi_actor/image_test/ROI.txt', 'r+')
f.truncate(0)

image, contours, hierarchy =  cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for t in range (0,3,1):
    if len(contours)>0:
        print('length is: ',len(contours))
        cnt = contours[t]
        x,y,w,h = cv2.boundingRect(cnt)
        draw_rec_lit_image=cv2.rectangle(rgb_img,(x,y),(x+w,y+h),(0,255,0),0)
        print('info of ',t,' contour ',x,',',y,',',x+w,',',y+h)
        x=x
        y=y
        x_1=x+w
        y_1=y+h
        
        f.write(str(x)+' ')
        f.write(str(y)+' ')
        f.write(str(x_1)+' ')
        f.write(str(y_1)+' ')
        f.write("\n")
        
        test_rgb = cv2.bitwise_and(draw_rec_lit_image, draw_rec_lit_image, mask=mask)
f.close()
## Display
cv2.imshow("mask", mask)
cv2.imshow("croped", croped)
cv2.imshow('lit_ROI', draw_rec_lit_image)
cv2.imshow("test_rgb", test_rgb)

cv2.imwrite('F:/unreal_cv_documentation/detect_ROI_for_multi_actor/image_test/desired_1.png',draw_rec_lit_image)
cv2.imwrite('F:/unreal_cv_documentation/detect_ROI_for_multi_actor/image_test/desired_2.png',test_rgb)

if cv2.waitKey() == ord('q'): #press q to close the output image window
        cv2.destroyAllWindows()
