# This file can be used to generate csv file for training and testing purpose
# Press 1 for training and any other number for testing file creation
# To execute this code please make a folder structure as like as follows--
# Images_folder
#  -- Train_folder
#  ---- Folder_1 (images of class 1 will be here)
#  ----**
#  ----**
#  ----**
#  ---- Folder_n (images of class n will be here)
#  -- Test_folder
#  ---- Folder_1 (images of class 1 will be here)
#  ----**
#  ----**
#  ----**
#  ---- Folder_n (images of class n will be here)

import os

print('*'*50)
print("Press 1 for the creation of Training CSV file \nPress 2 for the creation of Testing CSV file")
print('*'*50)

purpose_of_csv_file = input("Desire method: ")
purpose_of_csv_file = int(purpose_of_csv_file)
print(purpose_of_csv_file,'  ',type(purpose_of_csv_file))


current_dir = os.getcwd()
print(current_dir)


if purpose_of_csv_file == 1:
    print('CSV file for training will be generated')
    dirName = '/home/atif/machine_learning_stuff/ml_image/train_image_AI_better_resolution/image_container/crop/'
    file_name='train_image_file.csv'
else:
    print('CSV file for testing will be generated')
    dirName = '/home/atif/machine_learning_stuff/ml_image/train_image_AI_reduced/new_method/'
    file_name = 'test_image_file.csv'



csv_file_path = current_dir+'/'+file_name
if not os.path.isfile(csv_file_path):
    f = open(csv_file_path,'a')
    f.close()
    print('csv file is created')
else:
    os.remove(csv_file_path)
    f = open(csv_file_path,'a')
    f.close()
    print('old csv file is removed and new is created')


folder_list = []
ent = os.listdir(dirName)
for t in ent:
    folder_list.append(t)

print(folder_list)
folder_list.sort() # sorted folder name which is containing all of the images and the name of the folder is nothing but representative of class
print(folder_list)


f = open(csv_file_path, 'a')
f.write('filename'+';'+'classID')
f.write('\n')
for x in ent:
    # count = 0
    # print(x,'*'*20)
    class_id = folder_list.index(x) # use the index of the sorted folder list to define class number
    print(class_id,'*'*10,x)
    full_path = os.path.join(dirName,x)
    # print('folder_name: ',full_path)
    if os.path.isdir(full_path):
        new_dir = os.listdir(full_path)
        for y in new_dir:
            full_image_name_with_path = full_path+'/'+y
            # print(full_image_name_with_path,'#'*10,class_id)
            f.write(str(full_image_name_with_path))
            f.write(';')
            f.write(str(class_id))
            f.write('\n')
            # count+=1
            # print(y)
        # print('/\/'*40+'count of image: ',count)
        # count=1
        # print('!'*60)
