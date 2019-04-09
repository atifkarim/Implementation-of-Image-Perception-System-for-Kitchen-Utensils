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


#cropping image function

#path_mask = 'F:/save_image_ai/object_subtraction_for_UE4/image_AI/mask_calgonit'
#path_rgb = 'F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_calgonit'


def cropping_image(path_mask,path_rgb,dirname,image_number,lit,mask):
#    print('path_mask: ',path_mask,'\npath_rgb: ',path_rgb,'\ndirname: ',dirname)
    for a,b,image_mask in os.walk(path_mask):
        print('\na: ',a,'\tb: ',b,'\timage_mask: ',image_mask)
#        for s in image_mask:
#            print('\ns: ',s)
        n=image_mask[2].split("_")
#        print('\nn is: ',n)
        mask = path_mask+image_mask[mask]
#        print('\nmask is: ',mask)
#        print('\nimage_mask: ',image_mask[mask])
        image_mask2=cv2.imread(mask)
        
        hsv = cv2.cvtColor(image_mask2, cv2.COLOR_BGR2HSV)
        hsv_channels = cv2.split(hsv)
    
        _,thresh=cv2.threshold(hsv_channels[1],140,255,cv2.THRESH_BINARY_INV)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_mask2, contours, 0, (0,255,0), 3)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        
        for c,d,image_rgb in os.walk(path_rgb):
            print('\tc: ',c,'\td: ',d,'\timage_rgb: ',image_rgb)
#        for t in image_rgb:
            m=image_rgb[0].split('_')
#            print('\nval of m: ',m)
            if str(n[0])== str(m[0]):
#                print('OK')
                rgb = path_rgb+image_rgb[lit]
#                print('OK_1')
                image_rgb2=cv2.imread(rgb)
                image_rgb_rec=cv2.rectangle(image_rgb2,(x,y),(x+w,y+h),(255,255,255),1)
                crop_img = image_rgb_rec[y:y+h, x:x+w]
                cropped='cropped'
#                cv2.imwrite("F:/unreal_cv_documentation/ignore_from_git/YOLO_learning/BBox-Label-Tool/Images/cropped_image/jpg/test/"+str(t),crop_img)
                cv2.imwrite("F:/unreal_cv_documentation/my_crop/"+str(cropped)+str(n[0])+str(crop_1)+'.'+str(image_type),crop_img)
#                crop_1=crop_1+1
            else:
                pass
#        return 

crop_1=0

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
    index_lit=0
    index_mask=2
#    print('i is: ',i)
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
    for polar_angle in range(polar_angle_start,polar_angle_end,-10):
    
#        calculating the pitch value for different polar angle
        pitch=(180-(90+polar_angle))*(-1) #rotation around the y axis(you can denote by alpha)
    
        yaw=180 #rotaion around the z-axis(you can denote by 'beta')
        roll=0 #rotaion around x-axis(you can denote by 'gamma')
        for azimuthal_angle in range(azimuthal_angle_start,azimuthal_angle_end,1):
            print('I AM HERE!!!')
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
            
            res_lit = client.request('vget /camera/0/'+str(viewmode_1)+str(" ")+str(dirName)+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_1)+'.'+str(image_type)+'')
            res_mask = client.request('vget /camera/0/'+str(viewmode_2)+str(" ")+str(dirName)+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_2)+'.'+str(image_type)+'')
            res_normal = client.request('vget /camera/0/'+str(viewmode_3)+str(" ")+str(dirName)+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_3)+'.'+str(image_type)+'')
            crop_1+=1
            print('\ncrop_1: ',crop_1,'\tindex_lit: ',index_lit,'\tindex_mask: ',index_mask)
            cropping_image(path_mask=dirName,path_rgb=dirName,dirname=dirName,image_number=crop_1,lit=index_lit,mask=index_mask)
            index_lit=index_lit+3
            index_mask=index_mask+3
            print('\nJETZT crop_1: ',crop_1,'\tindex_lit: ',index_lit,'\tindex_mask: ',index_mask)
            print('I AM GOING')
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
#        print("polar_angle",polar_angle,"\z:",z,"\tpitch:",pitch,"\n")
        
