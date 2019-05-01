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
#import glob
import cv2
crop=0



def do_crop(path_of_image,lit_image_name,mask_image_name,crop_image_type):
#    print("function_start!!!!!!!!!!!!")
    global crop
    split_lit_image_name=lit_image_name.split(".")
    cropped="cropped"
    
    lit_image=path_of_image+lit_image_name
    mask_image=path_of_image+mask_image_name
    read_lit_image=cv2.imread(lit_image)
    read_mask_image=cv2.imread(mask_image)
    
    hsv = cv2.cvtColor(read_mask_image, cv2.COLOR_BGR2HSV)
    hsv_channels = cv2.split(hsv)
    _,thresh=cv2.threshold(hsv_channels[1],140,255,cv2.THRESH_BINARY_INV)
    im2,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(read_mask_image, contours, 0, (0,255,0), 3)
    cnt = contours[0]
#    print(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    
    draw_rec_lit_image=cv2.rectangle(read_lit_image,(x,y),(x+w,y+h),(255,255,255),1)
    crop_img = draw_rec_lit_image[y:y+h, x:x+w]
    crop=crop+1
#    print("crop: ",crop)
    cv2.imwrite(str(path_of_image)+str(split_lit_image_name[0])+"_"+str(cropped)+"."+str(crop_image_type),crop_img)
    


#config file is opening and fetching data
with open('config_file_capture_image.json', 'r') as f:
    config = json.load(f)

polar_angle_start= config['DEFAULT']['polar_angle_start']
polar_angle_end = config['DEFAULT']['polar_angle_end']
polar_angle_step = config['DEFAULT']['polar_angle_step']
azimuthal_angle_start= config['DEFAULT']['azimuthal_angle_start']
azimuthal_angle_end= config['DEFAULT']['azimuthal_angle_end']
azimuthal_angle_step= config['DEFAULT']['azimuthal_angle_step']
#print('pola_step: ',polar_angle_step,'\t azi step: ',azimuthal_angle_step)
viewmode_1= config['DEFAULT']['viewmode_1']
viewmode_2= config['DEFAULT']['viewmode_2']
viewmode_3= config['DEFAULT']['viewmode_3']
#address= config['DEFAULT']['address']
image_type= config['DEFAULT']['image_type']

#actor_list= config['DEFAULT']['actor_list']
#actor_list=client.request(actor_list)
#print("type is:",type(address))
print('actor: ',config['actor'])
actor_dict={}
for i in config['actor']:
    print(i)
    actor_dict[i]=[]
    actor_dict[i].append(polar_angle_start)
    actor_dict[i].append(polar_angle_end)
    actor_dict[i].append(azimuthal_angle_start)
    actor_dict[i].append(azimuthal_angle_end)
    actor_dict[i].append(config['actor'][i]['radius'])
print(actor_dict)


#print('type: ',type(actor_list),'\n list: ',actor_list)
#res_list=client.request('vget /objects')
#print('res_type: ',type(res_list),'res_list: ',res_list)

# Observing spherical movement of the camera around the object
# the area cover with polar_angle/elevation angle is 'Latitude' region. From north to South or vice versa
# the area cover with azimuthal angle is 'Longitude' region. From west to east or vice versa

for i in actor_dict:
    hide=client.request('vset /object/'+str(i)+'/hide')

for i in actor_dict:
    show=client.request('vset /object/'+str(i)+'/show')
    print('here: ',i)
    
    print("\nJOB_START")
    pic_num=1
    
#    creating directory to save present actor's captured image. here at firts create the folder written inside the inverted comma
#    for example here it is  F:/unreal_cv_documentation/my_dir/
    
    dirName='F:/unreal_cv_documentation/my_dir/'+str(i)+'/'
#    print(dirName)

    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")
        
#    getting the present actor's location
    actor_location=client.request('vget /object/'+str(i)+'/location')
    print(actor_location)
    actor_location = actor_location.split(" ") #splitted the location to append in an array
    actor_location_array=np.array(actor_location)
#    print(actor_location_array)
    actor_location_array = actor_location_array.astype(np.float) # make the string type to float type to use in the calculation

#    calculation process starts from here
    for polar_angle in range(polar_angle_start,polar_angle_end,polar_angle_step):
    
#        calculating the pitch value for different polar angle
        pitch=(180-(90+polar_angle))*(-1) #rotation around the y axis(you can denote by alpha)
    
        yaw=180 #rotaion around the z-axis(you can denote by 'beta')
        roll=0 #rotaion around x-axis(you can denote by 'gamma')
        for azimuthal_angle in range(azimuthal_angle_start,azimuthal_angle_end,azimuthal_angle_step):
#            print('I AM HERE!!!')
            centre_x=actor_location_array[0]      #centre of the object with respect to x-axis
            centre_y=actor_location_array[1]      #centre of the object with respect to y-axis
            centre_z=actor_location_array[2]      #centre of the object with respect to z-axis
            radius= actor_dict[i][-1]
            
#            Formula to find out the different points of x,y,z coordinates on the surface of a sphere is given below
#            print('radius is: ',radius)
            x= radius*(math.cos(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_x
            y= radius*(math.sin(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_y
            z= radius*(math.cos(math.radians(polar_angle)))+centre_z+15
            
#            setting camera location and rotation using the calculated value
            res_rotation=client.request('vset /camera/0/rotation '+str(pitch)+' '+str(yaw)+' '+str(roll)+'')
            res_location=client.request('vset /camera/0/location '+str(x)+' '+str(y)+' '+str(z)+'')
            yaw+=1 # yaw value is increasing to look at the object
            
            lit_name=str(pic_num)+"_"+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_1)+'.'+str(image_type)
            mask_name=str(pic_num)+"_"+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_2)+'.'+str(image_type)
            normal_name=str(pic_num)+"_"+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_3)+'.'+str(image_type)
            name_crop=str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)
            
            res_lit = client.request('vget /camera/0/'+str(viewmode_1)+str(" ")+str(dirName)+str(lit_name)+'')
            res_mask = client.request('vget /camera/0/'+str(viewmode_2)+str(" ")+str(dirName)+str(mask_name)+'')
            res_normal = client.request('vget /camera/0/'+str(viewmode_3)+str(" ")+str(dirName)+str(normal_name)+'')
            
            do_crop(path_of_image=dirName,lit_image_name=lit_name,mask_image_name=mask_name,crop_image_type=image_type)
            
#             if you want to use address info from config file then please use the following line
#            res = client.request('vget /camera/0/'+str(camera_view_type)+str(" ")+str(address)+str(pic_num)+'.'+str(image_type)+'')
        
# =============================================================================
#             old code , you can ignore it
#            res = client.request('vget /camera/0/viewmode_1 address'+str(pic_num)+'.png')
#             res = client.request('vget /camera/0/lit F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_table_4_21/'+str(pic_num)+'.png')
# =============================================================================
            
#            print("here dirName res is: ",res)
            pic_num+=1
#        print("for polar angle ",polar_angle," crop finish ",pic_num," times")
    crop=0
    print("\nCropping_is_finish_for ",i," actor")
    hide=client.request('vset /object/'+str(i)+'/hide')
    
print("\tJOB_DONE")
        
for i in actor_dict:
    show_again=client.request('vset /object/'+str(i)+'/show')
