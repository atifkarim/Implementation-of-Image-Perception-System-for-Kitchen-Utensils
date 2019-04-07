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

with open('config_file_capture_image.json', 'r') as f:
    config = json.load(f)
# getting the actor name and modify them to be used in python code
actor_name= config['DEFAULT']['actor_name']
#print("actor name type: ",type(actor_name))
#print("actors are: ",actor_name)
my_actor = actor_name.split(",")
#print("my_actor type after splitting: ",type(my_actor),"\nmy_actor after splitting: ",my_actor)
my_actor_array=np.array(my_actor)
#print("my_actor_array type: ",type(my_actor_array))
#print("x_actor_array: ",x_actor_array[1])

# =============================================================================
# for i in my_actor_array:
#     print(i)
# my_actor=str(my_actor_array[0])
# print('my_actor type: ',type(my_actor))
# =============================================================================

# =============================================================================
# actor_location=client.request('vget /object/'+str(my_actor)+'/location')
# print('actor_location: ',actor_location)
# print('actor_loaction_type: ',type(actor_location))
# actor_location = actor_location.split(" ")
# actor_location=np.array(actor_location)
# =============================================================================
# =============================================================================
# for j in actor_location:
#     print("hi: ",j)
# =============================================================================


polar_angle_start= config['DEFAULT']['polar_angle_start']
#print("polar angle start type: ",type(polar_angle_start))
polar_angle_end = config['DEFAULT']['polar_angle_end']
azimuthal_angle_start= config['DEFAULT']['azimuthal_angle_start']
azimuthal_angle_end= config['DEFAULT']['azimuthal_angle_end']
camera_view_type= config['DEFAULT']['camera_view_type']
address= config['DEFAULT']['address']
image_type= config['DEFAULT']['image_type']
#print("type is:",type(address))


# Observing spherical movement of the camera around the object

import math

# =============================================================================
# polar_angle_start = int(input('Enter polar_angle_start value:'))
# polar_angle_end = int(input('Enter polar_angle_end value:'))
# 
# azimuthal_angle_start = int(input('Enter azimuthal_angle_start value:'))
# azimuthal_angle_end = int(input('Enter azimuthal_angle_end value:'))
# =============================================================================

# the area cover with polar_angle/elevation angle is 'Latitude' region. From north to South or vice versa
# the area cover with azimuthal angle is 'Longitude' region. From west to east or vice versa

pic_num=1

for i in my_actor_array:
    
    #creating directory to save present actor's captured image
    
    dirName='F:/unreal_cv_documentation/my_dir/'+str(i)+'/'
    print(dirName)

    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
    
    actor_location=client.request('vget /object/'+str(i)+'/location')
    print("here location: ",actor_location)
    actor_location = actor_location.split(" ")
    actor_location_array=np.array(actor_location)
    actor_location_array = actor_location_array.astype(np.float)
    for j in actor_location_array:
        print("loc now: ",j)

    for polar_angle in range(polar_angle_start,polar_angle_end,-80):
    
    #calculating the pitch value for different polar angle
        pitch=(180-(90+polar_angle))*(-1) #rotation around the y axis(you can denote by alpha)
    
        yaw=180 #rotaion around the z-axis(you can denote by 'beta')
        roll=0 #rotaion around x-axis(you can denote by 'gamma')
        for azimuthal_angle in range(azimuthal_angle_start,azimuthal_angle_end,1):
        
            centre_x=actor_location_array[0]    #centre of the object with respect to x-axis
            centre_y=actor_location_array[1]      #centre of the object with respect to y-axis
            centre_z=actor_location_array[2]     #centre of the object with respect to z-axis
            radius=200.000      #randomly choosen a distance betwen object and camera
                            #from where the object is clearly visible
        
            #Formula to find out the different points of x,y,z coordinates on the surface of a sphere is given below
            x= radius*(math.cos(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_x
            y= radius*(math.sin(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_y
            z= radius*(math.cos(math.radians(polar_angle)))+centre_z+15
        
            res_rotation=client.request('vset /camera/0/rotation '+str(pitch)+' '+str(yaw)+' '+str(roll)+'')
            res_location=client.request('vset /camera/0/location '+str(x)+' '+str(y)+' '+str(z)+'')
            yaw+=1 # yaw value is increasing to look at the object
            
            res = client.request('vget /camera/0/'+str(camera_view_type)+str(" ")+str(dirName)+str(pic_num)+'.'+str(image_type)+'')
        
            #Comment out the following line to save image
            #        res = client.request('vget /camera/0/camera_view_type address'+str(pic_num)+'.png')
            #res = client.request('vget /camera/0/'+str(camera_view_type)+str(" ")+str(address)+str(pic_num)+'.'+str(image_type)+'')
            #res = client.request('vget /camera/0/lit F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_table_4_21/'+str(pic_num)+'.png')
            print("here dirName res is: ",res)
            pic_num+=1
        print("polar_angle",polar_angle,"\z:",z,"\tpitch:",pitch,"\n")
        
        
