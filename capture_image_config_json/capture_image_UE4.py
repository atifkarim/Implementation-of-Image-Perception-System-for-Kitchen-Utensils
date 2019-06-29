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
print('status: ',res)

import json
import numpy as np
import os
import math
import shutil
#import glob
import cv2
import copy
crop=0

#hide_Floor=client.request('vset /object/Floor/hide')
#hide_Floor_14=client.request('vset /object/Floor_14/hide')

def do_annotation(path_of_text_file,split_lit_image_name_0,x_1,y_1,x_2,y_2,class_number):
    
    f = open(path_of_text_file+str(split_lit_image_name_0)+'.txt', 'w')
    f.write('00'+str(class_number)+"\n")
    f.write(str(x_1)+' ')
    f.write(str(y_1)+' ')
    f.write(str(x_2)+' ')
    f.write(str(y_2)+' ')
    f.close()    

    print('Pre Annotation File has made !!')
    
    



def do_crop(path_of_RGB,path_of_MASK,path_of_CROP,lit_image_name,mask_image_name,crop_image_type,class_number,path_of_TEXT,red,green,blue):
#    print("function_start!!!!!!!!!!!!")
    global crop
    split_lit_image_name=lit_image_name.split(".")
    split_lit_image_name_0 = split_lit_image_name[0]
    cropped="cropped"
    
    lit_image=path_of_RGB+lit_image_name
    mask_image=path_of_MASK+mask_image_name
    read_lit_image=cv2.imread(lit_image)
    read_mask_image=cv2.imread(mask_image)
    img = np.zeros((read_mask_image.shape[0],read_mask_image.shape[1],read_mask_image.shape[2]), np.uint8)
    test_copy_img = copy.copy(img)
    
    for rows in range (0, read_mask_image.shape[0]):
        for cols in range(0, read_mask_image.shape[1]):
            if read_mask_image[rows][cols][0]==blue and read_mask_image[rows][cols][1]==green and read_mask_image[rows][cols][2]==red:
                test_copy_img[rows][cols] = 255 #for white color 255
                
    
    imgray=cv2.cvtColor(test_copy_img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    
   # image_mask_copy = read_mask_image.copy()
#    imgray=cv2.cvtColor(read_mask_image,cv2.COLOR_BGR2GRAY)
#    ret,thresh = cv2.threshold(imgray,127,255,1)
#    print('Till now OK')
#    image, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image_mask_copy,contours,0,(0,255,0))
    
#    hsv = cv2.cvtColor(read_mask_image, cv2.COLOR_BGR2HSV)
#    hsv_channels = cv2.split(hsv)
#    _,thresh=cv2.threshold(hsv_channels[1],140,255,cv2.THRESH_BINARY_INV)
#    im2,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(read_mask_image, contours, 0, (0,255,0), 3)
    if len(contours)>0:
        print('length is: ',len(contours))
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
    
        draw_rec_lit_image=cv2.rectangle(read_lit_image,(x,y),(x+w,y+h),(255,255,255),1)
        crop_img = draw_rec_lit_image[y:y+h, x:x+w]
        crop=crop+1
        cv2.imwrite(str(path_of_CROP)+str(split_lit_image_name[0])+"_"+str(cropped)+"."+str(crop_image_type),crop_img)
        print('CROP done')
        x_1=x
        y_1=y
        x_2=x+w
        y_2=y+h
        do_annotation(path_of_TEXT,split_lit_image_name_0,x_1,y_1,x_2,y_2,class_number)
        
    else:
        print('here length is: ',len(contours))
        print('-----NO Crop has done. No text file has made-----')
        pass


#config file is opening and fetching data
with open('test_config.json', 'r') as f:
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
viewmode_crop= config['DEFAULT']['viewmode_crop']
#address= config['DEFAULT']['address']
image_type= config['DEFAULT']['image_type']

#actor_list= config['DEFAULT']['actor_list']
#actor_list=client.request(actor_list)
#print("type is:",type(address))
#print('actor: ',config['actor'])
actor_dict={}
for i in config['actor']:
#    print(i)
    actor_dict[i]=[]
    actor_dict[i].append(polar_angle_start)
    actor_dict[i].append(polar_angle_end)
    actor_dict[i].append(azimuthal_angle_start)
    actor_dict[i].append(azimuthal_angle_end)
    actor_dict[i].append(config['actor'][i]['class'])
    actor_dict[i].append(config['actor'][i]['radius'])
print('actor_dict: ',actor_dict)

ind= {k_1:i_1 for i_1,k_1 in enumerate(actor_dict.keys())}

for u_1 in actor_dict:
    print('index of ',u_1,' is: ',ind[u_1])
    
#print('type: ',type(actor_list),'\n list: ',actor_list)
#res_list=client.request('vget /objects')
#print('res_type: ',type(res_list),'res_list: ',res_list)

# Observing spherical movement of the camera around the object
# the area cover with polar_angle/elevation angle is 'Latitude' region. From north to South or vice versa
# the area cover with azimuthal angle is 'Longitude' region. From west to east or vice versa
store=[]
print('my type is: ', type(store))
   
for i in actor_dict:
    RGB_info = 'RGB_info'
    sub_dir = str(RGB_info)
    dirName_RGB_info = 'F:/unreal_cv_documentation/my_dir/'+sub_dir+'/'
    
    if not os.path.exists(dirName_RGB_info):
        
        os.mkdir(dirName_RGB_info)
        print('Directory ',dirName_RGB_info,' created')
    else:
        pass
    f = open(str(dirName_RGB_info)+'color_info_'+str(i)+'.txt', 'a')
    f = open(str(dirName_RGB_info)+'color_info_'+str(i)+'.txt', 'r+')
    f.truncate(0)
    get_mask_color= client.request('vget /object/'+str(i)+'/color')
    print('mask color: ',get_mask_color)
    ww = str(get_mask_color)
    ww=ww.replace('R','')
    ww=ww.replace('G','')
    ww=ww.replace('B','')
    ww=ww.replace('A','')
    ww=ww.replace('=','')
    ww=ww.replace('(','')
    ww=ww.replace(')','')
    ww=ww.replace(',',' ')
    f.write(ww)
    f.write("\n")
    f.close()
    
for i in actor_dict:   
    with open(str(dirName_RGB_info)+'color_info_'+str(i)+'.txt') as f1:
        all_lines = f1.readlines()
        for line in all_lines:
            line_split = line.split('\n')
            m_1 = line_split[0].split(' ')
            print('DONE')
            store.append(m_1)
store=np.array(store,dtype=int)

r,g,b,a=[store[:,d] for d in range(len(store[0]))]
print(r,'\t',g,'\t',b,'\t',a)
print(r.shape,'\t',g.shape,'\t',b.shape,'\t',a.shape)
for i in actor_dict:
#    print('i is: ',i)
    hide=client.request('vset /object/'+str(i)+'/hide')
#    print('hidden actor: ',i,'\t',hide)
#number_1 = 0
for i in actor_dict:
    r_1 = r[ind[i]]
    g_1 = g[ind[i]]
    b_1 = b[ind[i]]
    
#    number_1 +=1
    print(r_1,'\t',g_1,'\t',b_1)
    object_class=actor_dict[i][-2]
    
    print('Working with ',i,' whose class is ',object_class,' has started')
    show=client.request('vset /object/'+str(i)+'/show')
    
#    RGB_info = 'RGB_info'
#    sub_dir = str(RGB_info)
#    dirName_RGB_info = 'F:/unreal_cv_documentation/my_dir/'+sub_dir+'/'
#    
#    if not os.path.exists(dirName_RGB_info):
#        
#        os.mkdir(dirName_RGB_info)
#        print('Directory ',dirName_RGB_info,' created')
#    else:
#        pass
#    f = open(str(dirName_RGB_info)+'color_info_'+str(i)+'.txt', 'a')
#    f = open(str(dirName_RGB_info)+'color_info_'+str(i)+'.txt', 'r+')
#    f.truncate(0)
#    get_mask_color= client.request('vget /object/'+str(i)+'/color')
#    print(get_mask_color)
#    ww = str(get_mask_color)
#    ww=ww.replace('R','')
#    ww=ww.replace('G','')
#    ww=ww.replace('B','')
#    ww=ww.replace('A','')
#    ww=ww.replace('=','')
#    ww=ww.replace('(','')
#    ww=ww.replace(')','')
#    ww=ww.replace(',',' ')
#    f.write(ww)
#    f.write("\n")
#    f.close()
#    
#    store=[]
#    with open(str(dirName_RGB_info)+'color_info_'+str(i)+'.txt') as f1:
#        a=f1.readlines()
#        for c in a:
#            k=c.split('\n')
#            m=k[0].split(' ')
#            store.append(m)
#    store=np.array(store,dtype=int)
#
#    r,g,b,a=[store[:,d] for d in range(len(store[0]))]
#    print(r,'\t',g,'\t',b,'\t',a)
    
#    set_mask_color= client.request('vset /object/'+str(i)+'/color 255 0 0')
#    print('set mask color: ',set_mask_color)
#    print('visible actor: ',i,'\t',show)
    
#    print("\nJOB_START")
    pic_num=1
    Labels='Labels'
#    num_1='00'
#    print(str(num_1))
#    text_num=1
    
#    creating directory to save present actor's captured image. here at firts create the folder written inside the inverted comma
#    for example here it is  F:/unreal_cv_documentation/my_dir/
    
#    dirName=r'F:/unreal_cv_documentation/my_dir/'+str(num_1)+str(object_class)+'/'+str(i)+'/'
    dirName_RGB='F:/unreal_cv_documentation/my_dir/'+str(viewmode_1)+'/'+'00'+str(object_class)+'/'
    dirName_MASK='F:/unreal_cv_documentation/my_dir/'+str(viewmode_2)+'/'+'00'+str(object_class)+'/'
    dirName_CROP='F:/unreal_cv_documentation/my_dir/'+str(viewmode_crop)+'/'+'00'+str(object_class)+'/'
    dirName_TEXT_FILE='F:/unreal_cv_documentation/my_dir/'+str(Labels)+'/'+'00'+str(object_class)+'/'
#    my_dir= 'F:/unreal_cv_documentation/my_dir/'+str(i)
#    print(dirName)
    
#    os.makedirs(dirName_RGB)
#    os.makedirs(dirName_MASK)
#    os.makedirs(dirName_CROP)
#    os.makedirs(dirName_TEXT_FILE)

    if not os.path.exists(dirName_RGB and dirName_MASK and dirName_CROP and dirName_TEXT_FILE):
        os.makedirs(dirName_RGB)
        os.makedirs(dirName_MASK)
        os.makedirs(dirName_CROP)
        os.makedirs(dirName_TEXT_FILE)
        print("Directory " , dirName_RGB,  " Created ")
        print("Directory " , dirName_MASK ,  " Created ")
        print("Directory " , dirName_CROP ,  " Created ")
        print("Directory " , dirName_TEXT_FILE ,  " Created ")
    else:
        pass
#        shutil.rmtree(my_dir,ignore_errors=True)
##        print('deleted old folder')
#        os.mkdir(dirName)
#        print('created new folder')
#        print("Directory " , dirName ,  " already exists")
        
#    getting the present actor's location
    
    actor_location=client.request('vget /object/'+str(i)+'/location')
    print(actor_location)
    actor_location = actor_location.split(" ") #splitted the location to append in an array
    actor_location_array=np.array(actor_location)
    print(actor_location_array)
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
#            object_class=actor_dict[i][-2]
#            print('here object_class: ',object_class,' and type: ',type(object_class))
            
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
            
            res_lit = client.request('vget /camera/0/'+str(viewmode_1)+str(" ")+str(dirName_RGB)+str(lit_name)+'')
            res_mask = client.request('vget /camera/0/'+str(viewmode_2)+str(" ")+str(dirName_MASK)+str(mask_name)+'')
            #res_normal = client.request('vget /camera/0/'+str(viewmode_3)+str(" ")+str(dirName_NORMAL)+str(normal_name)+'')
            #do_annotation(dirName,mask_name,lit_name,object_class)
            do_crop(path_of_RGB=dirName_RGB,path_of_MASK=dirName_MASK,path_of_CROP=dirName_CROP,lit_image_name=lit_name,mask_image_name=mask_name,crop_image_type=image_type,class_number=object_class, path_of_TEXT=dirName_TEXT_FILE,red=r_1,green=g_1,blue=b_1)
            
#             if you want to use address info from config file then please use the following line
#            res = client.request('vget /camera/0/'+str(camera_view_type)+str(" ")+str(address)+str(pic_num)+'.'+str(image_type)+'')
        
# =============================================================================
#             old code , you can ignore it
#            res = client.request('vget /camera/0/viewmode_1 address'+str(pic_num)+'.png')
#             res = client.request('vget /camera/0/lit F:/save_image_ai/object_subtraction_for_UE4/image_AI/rgb_table_4_21/'+str(pic_num)+'.png')
# =============================================================================
            #hide=client.request('vset /object/'+str(i)+'/hide')
            #res_lit_only_env = client.request('vget /camera/0/'+str(viewmode_1)+str(" ")+str(dirName)+str(pic_num)+"_env"+'.'+str(image_type))
#            print('name here: ',res_lit_only_env)
            #show=client.request('vset /object/'+str(i)+'/show')
#            print("here dirName res is: ",res)
            pic_num+=1
#        print("for polar angle ",polar_angle," crop finish ",pic_num," times")
    crop=0
#    print("\nCropping_is_finish_for ",i," actor")
    hide=client.request('vset /object/'+str(i)+'/hide')
    
print("\tJOB_DONE")
        
for i in actor_dict:
    show_again=client.request('vset /object/'+str(i)+'/show')
#    print('NOW visible actor: ',i,'\t',show_again)
#show_Floor=client.request('vset /object/Floor/show')
#show_Floor_14=client.request('vset /object/Floor_14/show')