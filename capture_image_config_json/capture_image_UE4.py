import time; print(time.strftime("The last update of this file: %Y-%m-%d %H:%M:%S", time.gmtime()))
import sys, time
# Establish connection with the UE4 game
from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)

# Checking status of connection between UnrealCV and UE4 game

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
#print(res)

import json
import numpy as np
import os
import math
import glob
import cv2

#config file is opening and fetching data
with open('config_file_capture_image.json', 'r') as f:
    config = json.load(f)

polar_angle_start= config['DEFAULT']['polar_angle_start']
polar_angle_end = config['DEFAULT']['polar_angle_end']
azimuthal_angle_start= config['DEFAULT']['azimuthal_angle_start']
azimuthal_angle_end= config['DEFAULT']['azimuthal_angle_end']

viewmode_1= config['DEFAULT']['viewmode_1']
viewmode_2= config['DEFAULT']['viewmode_2']
viewmode_3= config['DEFAULT']['viewmode_3']
#address= config['DEFAULT']['address']
image_type= config['DEFAULT']['image_type']
#print("type is:",type(address))

actor_dict={}
for i in config['actor']:
    actor_dict[config['actor'][i]['actor_name']]=[]
    actor_dict[config['actor'][i]['actor_name']].append(polar_angle_start)
    actor_dict[config['actor'][i]['actor_name']].append(polar_angle_end)
    actor_dict[config['actor'][i]['actor_name']].append(azimuthal_angle_start)
    actor_dict[config['actor'][i]['actor_name']].append(azimuthal_angle_end)
    actor_dict[config['actor'][i]['actor_name']].append(config['actor'][i]['radius'])

#global crop_1
#global index_lit
#global index_mask
#
#crop_1=0
#index_lit=0
#index_mask=2
#cropping image function

#path_mask = 'F:/save_image_ai/object_subtraction_for_UE4/image_AI/mask_calgonit'
#path_rgb = 'F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_calgonit'
#crop_1=0

def cropping_image(path_mask,path_rgb,dirname,naming_rule):
    print(path_mask)
    index_s=0
    index_t=1
    print("!!!!!!!!!FUNCTION CALL")
    
#    print('path_mask: ',path_mask,'\npath_rgb: ',path_rgb,'\ndirname: ',dirname)
    for a,b,image_mask in os.walk(path_mask):
        for s in image_mask:
#            print('\na: ',a,'\nb: ',b,'\nimage_mask: ',image_mask)
            if index_t<len(image_mask):
                print("here val in_t: ",index_t)
                print("started new cropping")
#                print('\ns: ',s)
                n=s.split("_")
#                print('\nn is: ',n)
#                print('\nn5 is: ',n[5])
                mask = path_mask+image_mask[index_s]
                print('\nmask is: ',mask)
#                print('\nimage_mask: ',image_mask[mask])
                image_mask2=cv2.imread(mask)
                
                hsv = cv2.cvtColor(image_mask2, cv2.COLOR_BGR2HSV)
                hsv_channels = cv2.split(hsv)
                
                _,thresh=cv2.threshold(hsv_channels[1],140,255,cv2.THRESH_BINARY_INV)
                im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_mask2, contours, 0, (0,255,0), 3)
                cnt = contours[0]
                x,y,w,h = cv2.boundingRect(cnt)
                print("till here OK_1")
                
                for c,d,image_rgb in os.walk(path_rgb):
#                    print('\nc: ',c,'\nd: ',d,'\nimage_rgb: ',image_rgb)
                    for t in image_rgb:
#                        print("t is: ",t)
                        m=t.split('_')
#                        print('\nval of m: ',m)
                        if int(n[0])== int(m[0]) and str(n[-1])!= str(m[-1]):
                            print("OK!!!")
                            rgb = path_rgb+image_rgb[index_t]
                            print("\nrgb: ",rgb)
                            #                print('OK_1')
                            image_rgb2=cv2.imread(rgb)
                            image_rgb_rec=cv2.rectangle(image_rgb2,(x,y),(x+w,y+h),(255,255,255),1)
                            crop_img = image_rgb_rec[y:y+h, x:x+w]
#                            print("crop_image: ",crop_img)
                            cropped='cropped'
                            img_type='.png'
                            #                cv2.imwrite("F:/unreal_cv_documentation/ignore_from_git/YOLO_learning/BBox-Label-Tool/Images/cropped_image/jpg/test/"+str(t),crop_img)
                            cv2.imwrite("F:/unreal_cv_documentation/my_crop/"+str(n[0])+"_"+str(naming_rule)+str(cropped)+str(img_type),crop_img)
                            print("CROP done")
                            index_t=index_t+2
                            index_s=index_s+2
#                            print("ok",len(image_mask))
                            print("\nin_s: ",index_s,"in_t: ",index_t)
#                        if index_t>len(image_mask):
#                            break
                        #                print('\nNOW image_rgb: ',image_rgb)
                        #                crop_1=crop_1+1
                        #                print("hi !!!!!!!!!!!!!!: ",crop_1)
                        
                        #                crop_1=crop_1+1
                        #                F:/unreal_cv_documentation/my_crop/
                        else:
                            pass
                        #        return 

def do_crop(path_of_image,lit_image_name,mask_image_name,crop_image_type):
    print("function_start!!!!!!!!!!!!")
    split_lit_image_name=lit_image_name.split(".")
    cropped="cropped"
    
    lit_image=path_of_image+lit_image_name
    mask_image=path_of_image+mask_image_name
    read_lit_image=cv2.imread(lit_image)
    read_mask_image=cv2.imread(mask_image)
    
    hsv = cv2.cvtColor(read_mask_image, cv2.COLOR_BGR2HSV)
    hsv_channels = cv2.split(hsv)
    _,thresh=cv2.threshold(hsv_channels[1],140,255,cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(read_mask_image, contours, 0, (0,255,0), 3)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    draw_rec_lit_image=cv2.rectangle(read_lit_image,(x,y),(x+w,y+h),(255,255,255),1)
    crop_img = draw_rec_lit_image[y:y+h, x:x+w]
    cv2.imwrite(str(path_of_image)+str(split_lit_image_name[0])+"_"+str(cropped)+"."+str(crop_image_type),crop_img)
    print("crop_done!!!!!!!!!")
#    print("lit: ",ip)
#    print("mask: ",op)
#    print("crop_type: ",crop_image_type)

# Observing spherical movement of the camera around the object

# =============================================================================
# polar_angle_start = int(input('Enter polar_angle_start value:'))
# polar_angle_end = int(input('Enter polar_angle_end value:'))
# 
# azimuthal_angle_start = int(input('Enter azimuthal_angle_start value:'))
# azimuthal_angle_end = int(input('Enter azimuthal_angle_end value:'))
# =============================================================================

# the area cover with polar_angle/elevation angle is 'Latitude' region. From north to South or vice versa
# the area cover with azimuthal angle is 'Longitude' region. From west to east or vice versa

for i in actor_dict:
#    crop_1=0
#    index_lit=0
#    index_mask=2
#    print('i is: ',i)
    print("\njourney_start")
    pic_num=1
    
#    creating directory to save present actor's captured image. here at firts create the folder written inside the inverted comma
#    for example here it is  F:/unreal_cv_documentation/my_dir/
    
    dirName='F:/unreal_cv_documentation/my_dir/'+str(i)+'/'
#    print(dirName)

    if not os.path.exists(dirName):
        os.mkdir(dirName)
#        print("Directory " , dirName ,  " Created ")
    else:
        hello=1
#        print("Directory " , dirName ,  " already exists")
        
#    getting the present actor's location
    actor_location=client.request('vget /object/'+str(i)+'/location')
#    print("here location: ",actor_location)
    actor_location = actor_location.split(" ") #splitted the location to append in an array
    actor_location_array=np.array(actor_location)
    actor_location_array = actor_location_array.astype(np.float) # make the string type to float type to use in the calculation

#    calculation process starts from here
    for polar_angle in range(polar_angle_start,polar_angle_end,-50):
    
#        calculating the pitch value for different polar angle
        pitch=(180-(90+polar_angle))*(-1) #rotation around the y axis(you can denote by alpha)
    
        yaw=180 #rotaion around the z-axis(you can denote by 'beta')
        roll=0 #rotaion around x-axis(you can denote by 'gamma')
        for azimuthal_angle in range(azimuthal_angle_start,azimuthal_angle_end,1):
#            print('I AM HERE!!!')
            centre_x=actor_location_array[0]      #centre of the object with respect to x-axis
            centre_y=actor_location_array[1]      #centre of the object with respect to y-axis
            centre_z=actor_location_array[2]      #centre of the object with respect to z-axis
            radius= actor_dict[i][-1]
            
#            if i == 'SM_SomatClassic_2':
#                radius=radius_somat
                
#            radius=250.000                        #randomly choosen a distance betwen object and camera
                                                    #from where the object is clearly visible
        
#            Formula to find out the different points of x,y,z coordinates on the surface of a sphere is given below
#            print('radius is: ',radius)
            x= radius*(math.cos(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_x
            y= radius*(math.sin(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_y
            z= radius*(math.cos(math.radians(polar_angle)))+centre_z+15
            
#            setting camera location and rotation using the calculated value
            res_rotation=client.request('vset /camera/0/rotation '+str(pitch)+' '+str(yaw)+' '+str(roll)+'')
            res_location=client.request('vset /camera/0/location '+str(x)+' '+str(y)+' '+str(z)+'')
            yaw+=1 # yaw value is increasing to look at the object
            
#            saving the image in the desired/created folder
#            res_lit = client.request('vget /camera/0/'+str(viewmode_1)+str(" ")+str(dirName)+str(pic_num)+'.'+str(image_type)+'')
#            res_mask = client.request('vget /camera/0/'+str(viewmode_2)+str(" ")+str(dirName)+str(pic_num)+'.'+str(image_type)+'')
            path_lit=str(pic_num)+"_"+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_1)+'.'+str(image_type)
            path_mask=str(pic_num)+"_"+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_2)+'.'+str(image_type)
            path_normal=str(pic_num)+"_"+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_3)+'.'+str(image_type)
            name_crop=str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)
            
            res_lit = client.request('vget /camera/0/'+str(viewmode_1)+str(" ")+str(dirName)+str(path_lit)+'')
            res_mask = client.request('vget /camera/0/'+str(viewmode_2)+str(" ")+str(dirName)+str(path_mask)+'')
            print("\npath_lit: ",path_lit)
            path_lit_split=path_lit.split(".")
            print("\npath_lit_split: ",path_lit_split)
            print("\npath_mask: ",path_mask)
            path_mask_split=path_mask.split(".")
            print("\npath_mask_split: ",path_mask_split)
            print("\nimage_type: ",image_type)
            print("i need this: ",path_lit_split[0],"\t",path_mask_split[0])
            
            do_crop(path_of_image=dirName,lit_image_name=path_lit,mask_image_name=path_mask,crop_image_type=image_type)
#            res_normal = client.request('vget /camera/0/'+str(viewmode_3)+str(" ")+str(dirName)+str(path_normal)+'')
#            cropping_image(path_mask=dirName,path_rgb=dirName,dirname=dirName,naming_rule=name_crop)

#            print('\ncrop_1: ',crop_1,'\nindex_lit: ',index_lit,'\nindex_mask: ',index_mask)
#            cropping_image(path_mask=dirName,path_rgb=dirName,dirname=dirName,image_number=crop_1,lit=index_lit,mask=index_mask)
            
#            print('\nJETZT crop_1: ',crop_1,'\nindex_lit: ',index_lit,'\nindex_mask: ',index_mask)
#            print('I AM GOING')
#            res = client.request('vget /camera/0/'+str(camera_view_type)+str(" ")+str(dirName)+str(pic_num)+'.'+str(image_type)+'')
            
#             if you want to use address info from config file then please use the following line
#            res = client.request('vget /camera/0/'+str(camera_view_type)+str(" ")+str(address)+str(pic_num)+'.'+str(image_type)+'')
        
# =============================================================================
#             old code , you can ignore it
#             res = client.request('vget /camera/0/camera_view_type address'+str(pic_num)+'.png')
#             res = client.request('vget /camera/0/lit F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_table_4_21/'+str(pic_num)+'.png')
# =============================================================================
            
#            print("here dirName res is: ",res)
            pic_num+=1
        print("hey --crop finish-- from function i am here also")
#    print("here dir is: ",dirName,"\npolar angle: ",polar_angle)
#    crop_1=0
#    cropping_image(path_mask=dirName,path_rgb=dirName,dirname=dirName,naming_rule=name_crop)
    print("\n",i," :this_actor_cropping_finish")
#    crop_1=crop_1+1
#    index_lit=index_lit+3
#    index_mask=index_mask+3
#        print("polar_angle",polar_angle,"\z:",z,"\tpitch:",pitch,"\n")
        
        
        
print("\tJOB DONE")
        
