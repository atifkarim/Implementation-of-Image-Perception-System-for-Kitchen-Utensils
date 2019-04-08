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

#config file is opening and fetching data
with open('config_file_capture_image.json', 'r') as f:
    config = json.load(f)
# getting the actor name and modify them to be used in python code
actor_name= config['DEFAULT']['actor_name']
#print("actor name type: ",type(actor_name))
#print("actors are: ",actor_name)
my_actor = actor_name.split(",") #careful !!! don't give any space after comma as there is no comma while you will print actor_name
                                 #if there is space after one actor name and then comma appears then give a space after the comma in the code
#print("my_actor type after splitting: ",type(my_actor),"\nmy_actor after splitting: ",my_actor)
my_actor_array=np.array(my_actor)
#print("my_actor_array type: ",type(my_actor_array))
#print("x_actor_array: ",x_actor_array[1])

polar_angle_start= config['DEFAULT']['polar_angle_start']
#print("polar angle start type: ",type(polar_angle_start))
polar_angle_end = config['DEFAULT']['polar_angle_end']
azimuthal_angle_start= config['DEFAULT']['azimuthal_angle_start']
azimuthal_angle_end= config['DEFAULT']['azimuthal_angle_end']
radius=config['DEFAULT']['radius']
radius_somat=config['SOMAT']['radius_somat']

viewmode_1= config['DEFAULT']['viewmode_1']
viewmode_2= config['DEFAULT']['viewmode_2']
viewmode_3= config['DEFAULT']['viewmode_3']

#address= config['DEFAULT']['address']
image_type= config['DEFAULT']['image_type']
#print("type is:",type(address))


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

for i in my_actor_array:
    pic_num=1
    
#    creating directory to save present actor's captured image. here at firts create the folder written inside the inverted comma
#    for example here it is  F:/unreal_cv_documentation/my_dir/
    
    dirName='F:/unreal_cv_documentation/my_dir/'+str(i)+'/'
    print(dirName)

    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
        
#    getting the present actor's location
    actor_location=client.request('vget /object/'+str(i)+'/location')
    print("here location: ",actor_location)
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
        
            centre_x=actor_location_array[0]      #centre of the object with respect to x-axis
            centre_y=actor_location_array[1]      #centre of the object with respect to y-axis
            centre_z=actor_location_array[2]      #centre of the object with respect to z-axis
            radius=radius
#            radius=250.000                        #randomly choosen a distance betwen object and camera
                                                    #from where the object is clearly visible
        
#            Formula to find out the different points of x,y,z coordinates on the surface of a sphere is given below
            x= radius*(math.cos(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_x
            y= radius*(math.sin(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_y
            z= radius*(math.cos(math.radians(polar_angle)))+centre_z+15
            
#            setting camera location and rotation using the calculated value
            res_rotation=client.request('vset /camera/0/rotation '+str(pitch)+' '+str(yaw)+' '+str(roll)+'')
            res_location=client.request('vset /camera/0/location '+str(x)+' '+str(y)+' '+str(z)+'')
            yaw+=1 # yaw value is increasing to look at the object
            
#            saving the image in the desired/created folder
            print("hi coder: azim>> ",azimuthal_angle," and polar>>> ",polar_angle,"\n")
            res = client.request('vget /camera/0/'+str(viewmode_1)+str(" ")+str(dirName)+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_1)+'.'+str(image_type)+'')
            res = client.request('vget /camera/0/'+str(viewmode_2)+str(" ")+str(dirName)+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_2)+'.'+str(image_type)+'')
            res = client.request('vget /camera/0/'+str(viewmode_3)+str(" ")+str(dirName)+str(i)+'_'+str(azimuthal_angle)+"_"+str(polar_angle)+'_'+str(viewmode_3)+'.'+str(image_type)+'')
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
        
        
