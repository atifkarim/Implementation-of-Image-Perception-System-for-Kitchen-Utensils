# -*- coding: utf-8 -*-

import os
from os import walk, getcwd
from PIL import Image


dirName = "/media/atif/BE0E05910E0543BD/University of Bremen MSc/problem_solving/convert_yolo_folder/Images/"


folder_name_string = []
ent = os.listdir(dirName)
for t in ent:
    folder_name_string.append(t)

print(folder_name_string)
folder_name_string.sort()
print(folder_name_string)


classes = []
for index, values in enumerate(folder_name_string):
#    print(index)
    index_1 = index+1
#    print(index_1)
    index_2 = "00"+str(index_1)
    print(index_2)
    classes.append(index_2)

print(classes)


#folder_name_string = ["calgonit_finish","calgonit_finish_klarspueler","calgonit_maschine_pfleger"]
#classes = ["001","002","003"]
#classes = ["001","002","003","004","005","006","007","008","009","0010","0011","0012","0013","0014","0015","0016","0017","0018","0019"]
#classes = ["001","002"]
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

print('folder_name \t class(number to be pressed)')
for idx,val in enumerate (folder_name_string):
    print(folder_name_string[idx],'\t\t',classes[idx])



    
    
"""-------------------------------------------------------------------""" 

""" Configure Paths"""
print("Give the folder number using 1 or 2 digit.")
print("For example, if the folder name is 001 then put 1, for 005 put 5 but if name is 0012 then put 12")
folder_number = input("folder_number: ")
print(folder_number)

folder_num_here = "00"+folder_number
#print(folder_num_here)

index_of_this_folder_im_classes = classes.index(folder_num_here)
#print(index_of_this_folder_im_classes)
main_folder_name = folder_name_string[index_of_this_folder_im_classes]
#print(type(main_folder_name),' ',main_folder_name)


mypath = "/media/atif/BE0E05910E0543BD/University of Bremen MSc/problem_solving/convert_yolo_folder/Labels/"+main_folder_name+"/"
outpath = "/media/atif/BE0E05910E0543BD/University of Bremen MSc/problem_solving/convert_yolo_folder/Labels/out/"+main_folder_name+"/"

#mypath = "/home/atif/machine_learning_stuff/YOLO_learning/YOLO-Annotation-Tool/Labels/00"+folder_number+"/"
#outpath = "/home/atif/machine_learning_stuff/YOLO_learning/YOLO-Annotation-Tool/Labels/out/00"+folder_number+"/"

#mypath = "/home/atif/machine_learning_stuff/YOLO_learning/YOLO-Annotation-Tool/Labels/001/"
#outpath = "/home/atif/machine_learning_stuff/YOLO_learning/YOLO-Annotation-Tool/Labels/out/001/"

if not os.path.exists(outpath):
    os.makedirs(outpath)
    print('created : ', outpath)
else:
    print(' has already created',outpath)
    pass

#print(mypath)
#print(outpath)

cls = "00"+folder_number # class or class number will be same as the folder number
#cls = "001"
#print(type(cls))

if cls not in classes:
    print("invalid class")
    exit(0)
cls_id = classes.index(cls)

wd = getcwd()
list_file = open('%s/%s_list.txt'%(wd, main_folder_name), 'w')
print(wd)
print(list_file)


""" Get input text file list """
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    txt_name_list.extend(filenames)
    print("break")
    break
# print(txt_name_list)

print("TILL HERE OKK")
""" Process """
for txt_name in txt_name_list:
    # txt_file =  open("Labels/stop_sign/001.txt", "r")
    
    """ Open input text files """
    txt_path = mypath + txt_name
    print("Input:" + txt_path)
    txt_file = open(txt_path, "r")
    lines = txt_file.read().split('\n')   #for ubuntu, use "\r\n" instead of "\n"
    
    """ Open output text files """
    txt_outpath = outpath + txt_name
    print("I am here Output:" + txt_outpath)
    txt_outfile = open(txt_outpath, "a")
#    print("iiiii")
    
    
    """ Convert the data to YOLO format """
    ct = 0
    for line in lines:
        #print('lenth of line is: ')
        #print(len(line))
        #print('\n')
        elems = line.split(' ')
        if(len(elems) >= 2):
        #if(len(line) >= 2):
            ct = ct + 1
            print(line + "\n")
            elems = line.split(' ')
            print(elems)
            xmin = elems[0]
            xmax = elems[2]
            ymin = elems[1]
            ymax = elems[3]
            print(elems[0])
            #
            img_path = str('%s/Images/%s/%s.png'%(wd, main_folder_name, os.path.splitext(txt_name)[0]))
#            img_path = str('%s/Images/%s/%s.png'%(wd, cls, os.path.splitext(txt_name)[0]))
            #t = magic.from_file(img_path)
            #wh= re.search('(\d+) x (\d+)', t).groups()
            im=Image.open(img_path)
            w= int(im.size[0])
            h= int(im.size[1])
            #w = int(xmax) - int(xmin)
            #h = int(ymax) - int(ymin)
            # print(xmin)
            print(w, h)
            print(float(xmin), float(xmax), float(ymin), float(ymax))
            b = (float(xmin), float(xmax), float(ymin), float(ymax))
            bb = convert((w,h), b)
            print(bb)
            txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    """ Save those images with bb into list"""
    if(ct != 0):
        print("ok")
        list_file.write('%s/images/%s/%s.png\n'%(wd, cls, os.path.splitext(txt_name)[0]))
                
list_file.close()