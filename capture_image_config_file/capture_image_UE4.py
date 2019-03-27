import time; print(time.strftime("The last update of this file: %Y-%m-%d %H:%M:%S", time.gmtime()))
import sys, time

# Establish connection with the UE4 game

from configparser import SafeConfigParser

parser = SafeConfigParser()
parser.read('capture_image_config.ini')

polar_angle_start=parser.get('capture_image', 'polar_angle_start')
polar_angle_start=int(polar_angle_start)

polar_angle_end=parser.get('capture_image', 'polar_angle_end')
polar_angle_end=int(polar_angle_end)

azimuthal_angle_start=parser.get('capture_image', 'azimuthal_angle_start')
azimuthal_angle_start=int(azimuthal_angle_start)

azimuthal_angle_end=parser.get('capture_image', 'azimuthal_angle_end')
azimuthal_angle_end=int(azimuthal_angle_end)

address=parser.get('capture_image', 'address')
print("address is : ",address)


from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)

# Checking status of connection between UnrealCV and UE4 game

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
print(res)

# Observing spherical movement of the camera around the object

import math

#polar_angle_start = int(input('Enter polar_angle_start value:'))
#polar_angle_end = int(input('Enter polar_angle_end value:'))

#azimuthal_angle_start = int(input('Enter azimuthal_angle_start value:'))
#azimuthal_angle_end = int(input('Enter azimuthal_angle_end value:'))

# the area cover with polar_angle/elevation angle is 'Latitude' region. From north to South or vice versa
# the area cover with azimuthal angle is 'Longitude' region. From west to east or vice versa

pic_num=1

for polar_angle in range(polar_angle_start,polar_angle_end,-10):
    
    #calculating the pitch value for different polar angle
    pitch=(180-(90+polar_angle))*(-1) #rotation around the y axis(you can denote by alpha)
    
    yaw=180 #rotaion around the z-axis(you can denote by 'beta')
    roll=0 #rotaion around x-axis(you can denote by 'gamma')
    for azimuthal_angle in range(azimuthal_angle_start,azimuthal_angle_end,1):
        
        centre_x=-45    #centre of the object with respect to x-axis
        centre_y=0      #centre of the object with respect to y-axis
        centre_z=32     #centre of the object with respect to z-axis
        radius=350      #randomly choosen a distance betwen object and camera
                        #from where the object is clearly visible
        
        #Formula to find out the different points of x,y,z coordinates on the surface of a sphere is given below
        x= radius*(math.cos(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_x
        y= radius*(math.sin(math.radians(azimuthal_angle)))*(math.sin(math.radians(polar_angle)))+centre_y
        z= radius*(math.cos(math.radians(polar_angle)))+centre_z+15
        
        res_rotation=client.request('vset /camera/0/rotation '+str(pitch)+' '+str(yaw)+' '+str(roll)+'')
        res_location=client.request('vset /camera/0/location '+str(x)+' '+str(y)+' '+str(z)+'')
        yaw+=1 # yaw value is increasing to look at the object
        
        #Comment out the following line to save image
# =============================================================================
#     	  res = client.request('vget /camera/0/lit F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_table/'+str(pic_num)+'.png')
#         res = client.request('vget /camera/0/lit F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_table_4_21/'+str(pic_num)+'.png')
        res = client.request('vget /camera/0/lit '+str(address)+str(pic_num)+'.png')
#        print("type of res: ",type(res))
#        print("address is now : ",address)
# =============================================================================
        pic_num+=1
print("polar_angle",polar_angle,"\z:",z,"\tpitch:",pitch,"\n")
        
        
