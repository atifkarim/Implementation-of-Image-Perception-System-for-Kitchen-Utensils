# In this code the approach of finding ROI of different coloured object using MASK image and LIT(RGB) image has depicted.
# To perform the operation of this code at first you have to follow this link and run the code. The code of this link will take the image from UE4 using UnrealCV and
# make a text file which will contain the RGBA info of object(from MASK image it will take it). It is important to fetch the RGB info of every actor.
# Keep in mind that UnealCV generates the color info in RGB format while OpenCV deals with BGR format.

#importing library
import cv2
import numpy as np
import copy

# desired actor list

actor_array = ['SM_DenkMitEdelstahlReinigerSpray_18','SM_CalgonitFinishKlarspueler_3','SM_CalgonitFinishVorratspack_12']
actor_array = np.array(actor_array)

#path and type of saved image
dirName = 'F:/unreal_cv_documentation/detect_ROI_for_multi_actor/new_image/'
image_type = '.png'

#load the image. Please change the name and directory in your case
rgb_image = cv2.imread(str(dirName)+'lit_new_2019-06-26_22-41-31'+str(image_type))
mask_image = cv2.imread(str(dirName)+'mask_new_2019-06-26_22-41-31'+str(image_type))

#from the text file where RGB info of desired are saved has fetched and store in an array to use later
import numpy as np
store=[]
with open(str(dirName)+'color_info2019-06-26_22-41-31.txt') as f1:
    a=f1.readlines()
    for i in a:
        k=i.split('\n')
        m=k[0].split(' ')
        store.append(m)
store=np.array(store,dtype=int)

r,g,b,a=[store[:,i] for i in range(len(store[0]))]

#following lines are the main ideas to detect ROI from MASK image using RGB info

img = np.zeros((mask_image.shape[0],mask_image.shape[1],mask_image.shape[2]), np.uint8) #create a black coloured image
test_copy_img = copy.copy(img) #here all desired object will be passed after converting to white color
test_copy_img_color = copy.copy(img) #here all desired object with their mask color will be passed

for in_actor,i in enumerate(actor_array):
    for rows in range (0, mask_image.shape[0]):
        for cols in range(0, mask_image.shape[1]):
            if mask_image[rows][cols][0]==b[in_actor] and mask_image[rows][cols][1]==g[in_actor] and mask_image[rows][cols][2]==r[in_actor]: #take a look in this line. BGR is the order
                test_copy_img[rows][cols] = 255 #for white color 255
                test_copy_img_color[rows][cols]=mask_image[rows][cols] #only mask coloured object has passed
                
test_copy_image_color_mask = copy.copy(test_copy_img_color) #tis image has copied to use for drawing ROI where the image will be blacked but the object will be in mask color
                
imgray=cv2.cvtColor(test_copy_img,cv2.COLOR_BGR2GRAY) #convert the image to grayscale where desired object are in white color and background is black
ret,thresh = cv2.threshold(imgray,127,255,0) #do thresholding
image, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #finding contour

roi_list = [] # a list to store the ROI info
rgb_image_copy = copy.copy(rgb_image)
for t in range (0,len(actor_array),1):
    if len(contours)>0:
        print('length is: ',len(contours))
        cnt = contours[t]
        x,y,w,h = cv2.boundingRect(cnt) #detect the boundary point
        roi_rgb_image=cv2.rectangle(rgb_image_copy,(x,y),(x+w,y+h),(0,255,0),1) #draw ROI on the RGB image
        roi_black_mask = cv2.rectangle(test_copy_image_color_mask,(x,y),(x+w,y+h),(0,255,0),1)
        print('info of ',t,' contour ',x,',',y,',',x+w,',',y+h)
        x=x
        y=y
        x_1=x+w
        y_1=y+h
        
        roi_point = [x,y,x_1,y_1]
        roi_list.append(roi_point) #storing ROI info in the list
    else:
        pass


# storing the ROI info in a text file

f = open(str(dirName)+'roi_info.txt', 'a')
f = open(str(dirName)+'roi_info.txt', 'r+')
f.truncate(0)

for i in roi_list:
    ww = str(i)
    ww=ww.replace('[','')
    ww=ww.replace(']','')
    ww=ww.replace(',','')
    f.write(ww)
    f.write("\n")
f.close()

# convert the RGB image to black color except the region of the desired object
# the idea is to compare the RGB image with one image where background while desired object are in the white color and replace RGB image's pixel with '0' value
# which is not representing the desired object's region

a_copy = copy.copy(test_copy_img)
rgb_black = copy.copy(rgb_image)

for rows in range (0, a_copy.shape[0]):
    for cols in range (0, a_copy.shape[1]):
        if a_copy[rows][cols][0]==0 and a_copy[rows][cols][1]==0 and a_copy[rows][cols][2]==0:
            rgb_black[rows][cols]=0
        else:
            pass

# draw ROI on the black colored RGB image

rgb_black_roi = copy.copy(rgb_black)

for i in roi_list:
    x_new = i[0]
    y_new = i[1]
    x_w_new = i[2]
    y_h_new = i[3]
    
    roi_black_rgb = cv2.rectangle(rgb_black_roi,(x_new,y_new),(x_w_new,y_h_new),(0,255,0),1)

# save the image if you want

# save the image if you want

cv2.imwrite(str(dirName)+'test_copy_image'+str(image_type),test_copy_img)
cv2.imwrite(str(dirName)+'test_copy_image_color'+str(image_type),test_copy_img_color)
cv2.imwrite(str(dirName)+'roi_rgb_image'+str(image_type),roi_rgb_image)
cv2.imwrite(str(dirName)+'rgb_black'+str(image_type),rgb_black)
cv2.imwrite(str(dirName)+'roi_black_rgb'+str(image_type),roi_black_rgb)
cv2.imwrite(str(dirName)+'roi_black_mask'+str(image_type),roi_black_mask)

# Display the image

cv2.imshow('rgb', rgb_image)
cv2.imshow('mask', mask_image)
cv2.imshow('test_copy_img',test_copy_img)
cv2.imshow('test_copy_img_color', test_copy_img_color)
cv2.imshow('roi_rgb_image', roi_rgb_image)
cv2.imshow('roi_black_mask',roi_black_mask)
cv2.imshow('rgb_black', rgb_black)
cv2.imshow('roi_black_rgb', roi_black_rgb)


if cv2.waitKey() == ord('q'): #press q to close the output image window
        cv2.destroyAllWindows()
