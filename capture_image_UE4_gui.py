# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:16:36 2019

@author: atif
"""

from tkinter import *

import time; print(time.strftime("The last update of this file: %Y-%m-%d %H:%M:%S", time.gmtime()))
import sys, time

import math

# Establish connection with the UE4 game

# =============================================================================
# from unrealcv import client
# client.connect()
# if not client.isconnected():
#     print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
#     sys.exit(-1)
# 
# # Checking status of connection between UnrealCV and UE4 game
# 
# res = client.request('vget /unrealcv/status')
# # The image resolution and port is configured in the config file.
# print(res)
# =============================================================================
root= Tk()
root.geometry('500x500+0+0')
label_name=Label(root,font=('arial',12,'bold'),bd=5 , text="UE4 image capture GUI").grid(row=0, column=5)
label_name=Label(root,font=('arial',12,'bold'),bd=5 , text="initial polar value").grid(row=1, column=1)
label_name=Label(root,font=('arial',12,'bold'),bd=5 , text=" ").grid(row=1, column=3)

label_name=Label(root,font=('arial',12,'bold'),bd=5 , text="final polar value").grid(row=3, column=1)
label_name=Label(root,font=('arial',12,'bold'),bd=5 , text=" ").grid(row=3, column=3)

label_name=Label(root,font=('arial',12,'bold'),bd=5 , text="final azimuthal value").grid(row=5, column=1)
label_name=Label(root,font=('arial',12,'bold'),bd=5 , text=" ").grid(row=5, column=3)

label_name=Label(root,font=('arial',12,'bold'),bd=5 , text="final azimuthal value").grid(row=7, column=1)
label_name=Label(root,font=('arial',12,'bold'),bd=5 , text=" ").grid(row=7, column=3)

initial_polar_value=StringVar()
initial_polar_value_entry=Entry(root,textvariable=initial_polar_value)
initial_polar_value_entry.grid(row=1,column=5)

final_polar_value=StringVar()
final_polar_value_entry=Entry(root,textvariable=final_polar_value)
final_polar_value_entry.grid(row=3,column=5)

initial_azimuthal_value=StringVar()
initial_azimuthal_value_entry=Entry(root,textvariable=initial_azimuthal_value)
initial_azimuthal_value_entry.grid(row=5,column=5)

final_azimuthal_value=StringVar()
final_azimuthal_value_entry=Entry(root,textvariable=final_azimuthal_value)
final_azimuthal_value_entry.grid(row=7,column=5)

def execute():
    polar_angle_start=int(initial_polar_value_entry.get())
    polar_angle_end=int(final_polar_value_entry.get())
    azimuthal_angle_start=int(initial_azimuthal_value_entry.get())
    azimuthal_angle_end=int(final_azimuthal_value_entry.get())
    pic_num=1
    for polar_angle in range(polar_angle_start,polar_angle_end,-30):
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
            #res = client.request('vget /camera/0/lit F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_table/'+str(pic_num)+'.png')
            #res = client.request('vget /camera/0/lit F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_table_4_21/'+str(pic_num)+'.png')
            pic_num+=1
        print("polar_angle",polar_angle,"\z:",z,"\tpitch:",pitch,"\n")
    
    
def close():
    root.destroy()
    exit()

btn_exe = Button(root,font=('arial',12,'bold'),bd=6 , text="EXECUTE ",command = execute)
btn_exe.grid(row =9 , column=5)

btn_exit = Button(root,font=('arial',12,'bold'),bd=6 , text="  EXIT  ",bg='red', command = close)
btn_exit.grid(row=11, column=5)


mainloop()