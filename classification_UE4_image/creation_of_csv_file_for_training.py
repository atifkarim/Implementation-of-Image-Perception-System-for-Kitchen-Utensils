import os


dirName = '/home/atif/machine_learning_stuff/ml_image/train_image_AI_better_resolution/image_container/crop/'
#dirName = '/home/atif/machine_learning_stuff/ml_image/train_image_AI_better_resolution/image_container/crop/'


current_dir = os.getcwd()
print(current_dir)
file_name = 'train_image_file.csv'

evaluation_metrics_file_path = current_dir+'/'+file_name
if not os.path.isfile(evaluation_metrics_file_path):
    f = open(evaluation_metrics_file_path,'a')
    f.close()
    print('csv file for training is created')
else:
    os.remove(evaluation_metrics_file_path)
    f = open(evaluation_metrics_file_path,'a')
    f.close()
    print('csv file for testing is created')


folder_list = []
ent = os.listdir(dirName)
for t in ent:
    folder_list.append(t)

print(folder_list)
folder_list.sort()
print(folder_list)


f = open(evaluation_metrics_file_path, 'a')
f.write('filename'+';'+'classID')
f.write('\n')
for x in ent:
    # count = 0
    # print(x,'*'*20)
    class_id = folder_list.index(x)
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
