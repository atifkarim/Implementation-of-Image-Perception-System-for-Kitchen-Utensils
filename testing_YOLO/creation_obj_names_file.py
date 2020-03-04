# Creation of obj.names is possibe with this code. It will take all of the classes name using the folder's name from the referred path

import os

dirName = '/media/atif/BE0E05910E0543BD/University of Bremen MSc/problem_solving/convert_yolo_folder/Images/'

current_dir = os.getcwd()
print(current_dir)
file_name = 'obj.names'

obj_names_file_path = current_dir+'/'+file_name
if not os.path.isfile(obj_names_file_path):
    f = open(obj_names_file_path,'a')
    f.close()
    print('file now created')
else:
    os.remove(obj_names_file_path)
    f = open(obj_names_file_path,'a')
    f.close()
    print('file removed and created')


folder_list = []
ent = os.listdir(dirName)
for t in ent:
    folder_list.append(t)

print(folder_list)
folder_list.sort()
print(folder_list)


f = open(obj_names_file_path, 'a')

for x in folder_list:
    print(x)
    f.write(str(x))
    f.write('\n')
f.close()
