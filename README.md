# Autonomous Image Perception System for Inventory Robot
The goal of this project is to develop a system which will be used in a retailer shop to count and filling objects from the shell. The project is conducting by [Institute for Artificial Intelligence](https://ai.uni-bremen.de/). Ideas and developed steps are discussed below.

# Requirements
* Unreal Engine (Tested in Windows 10, 64 bit machine with v4.16, 4.19)
* UnrealCV(fast branch)
* Python 3.6
* Windows 10, 64 bit


# Creation of image dataset from Unreal Engine
The goal of this part is to develop a system which will be capable of capturing image of object which isimported as an actor in UE4.  A brief introduction about instaling UE4, UnrealCV and basics of the codeis available [here](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/unreal_cv_documentaion.pdf).

Full code is available [here](https://github.com/atifkarim/unreal_cv_image_manipulation/tree/master/capture_UE4_image).
Step by step the working method of the automated image capturing tool is described below.

* First of all of the objects which will be captured by UE4 will be placed or imported in the game
* A [configuration file](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/config_file_capture_image.json) is made where all of the angles and objects name with the radius(distance between camera and the object) are mentioned.
* Before running the [main python file](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/capture_image_UE4.py) it is obvious to turn on the PLAY moodin UE4
* In [main.py](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/capture_image_UE4.py) two functions are called, one for cropping the object from the RGB image using **the mask information** and another is for doing the annotation
* After running the [main.py](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/capture_image_UE4.py) for one actoar at a time RGB image, mask image, normal image, ROI(ina text file) and Mask Image information(1 text file for one actor) are saved.
 Two things to remind --
 
 * [Here](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/d9c6e9af88067b8135d5ca100b13d9238dc6abba/capture_image_config_json/capture_image_UE4.py#L105) give the config file name that you want to use

* In the config file all minimal info has given which will control how to take the image like polar, azimuthal angle, viewmode of UE4, image type etc. Pay attention [here](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/d9c6e9af88067b8135d5ca100b13d9238dc6abba/capture_image_config_json/config_file_capture_image.json#L5), to give **minus**/**negative** sign before the step of polar angle.


After performing the above task the output folder structure will be as follows --
* image_container(folder of all image and ROI information. This will be created in the directory where [code](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/capture_image_UE4.py) is situated)
  * crop (content is cropped image)
  * Labels (content is ROI information of each position)
  * lit (content is RGB image)
  * object_mask (content is mask image)
  * RGB_info

**Sample image and ROI file is linked below**

![lit image](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/sample_image_and_label/1_SM_CalgonitFinish_2_0_90_60_lit.png)
![mask image](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/sample_image_and_label/1_SM_CalgonitFinish_2_0_90_60_object_mask.png)
![cropped image](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/sample_image_and_label/1_SM_CalgonitFinish_2_0_90_60_lit_cropped.png)


## [ROI information](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/sample_image_and_label/1_SM_CalgonitFinish_2_0_90_60_lit.txt)
## [Color Mask Info](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/sample_image_and_label/color_info_SM_CalgonitFinish_2.txt)

# Object detetion and localization using YOLO

This section will show the method of using YOLO for detecting object from the Image and Video.  It will cover the top to bottom process of making data-set using YOLO and the way to use them like trainingand testing.To know about the theory of the YOLO [this paper](https://arxiv.org/abs/1506.02640) & [this link](https://pjreddie.com/darknet/yolo/) will be a good one to start.
\
Training with YOLO can be divided into two parts.  First one is to convert dataset for YOLO processand the later will stand for the training.

**Dataset for YOLO**
To convert dataset for YOLO means to make dataset containing the ROI(region of interest) informationwith class number.  A structured repository for this operation could be found [here](https://github.com/ManivannanMurugavel/YOLO-Annotation-Tool).  In this repository two folders and one file is most important and they are as follows:
* Images
* Labels
* convert.py

Here, **Images** folder keep all of the RGB images with sub-folder(sub-folder means the folder which will contains images of different class. Naming of folders could be anything eg. class name).  In the Labels folder keep the sub-folders which contains all the ROI information of actors in each image.  Inside of the Labels folder a destination folder named **out** should be created where the **converted YOLO format** file  is  saved.   Reason  of  using *convert.py* file is to  generate files in YOLO format. I have modified the *convert.py* file and it would be found [here](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/convert_yolo_dataset_modified.py).


# Arrangement of folder and files to train with YOLO

The contents of the out folder (text files with ROI information) and the Images folder would be placed in a folder named obj in the darknet/data/directory.  This darknet folder is nothing but a [git repository](https://github.com/pjreddie/darknet) which is also  mentioned  in  [this  link](https://pjreddie.com/darknet/yolo/). During  putting  all  of  the  images  and  text  file  don’t  use  any sub-directory. Direct put all of the contents.\
Another two files named obj.names and obj.data these two files also be placed in the darknet/data/directory. \
Content of obj.data is like as the following lines -\
classes= 19\
train = data/train.txt\
valid = data/test.txt\
names = data/obj.names\
backup = backup/

Description of these lines are -
* *classes* is the value of used class to train.  In this project used class was **19**. 
* *train* and *valid* are two files which contains the image path of training and validation images.  To make these files one python file is used named [process.py](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/process.py) which will be placed in the darknet/data/directory. 
* *backup* is the folder name where trained weights will be store and whose path is darknet/backup/.
* *names* is  the  variable  which  indicates  the  path  of obj.names whose  contained  information  are  the  name  of  all classes.  \

This [code](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/creation_obj_names_file.py) can be used to create this obj.names file. \
In darknet/cfg/directory yolo-obj.cfg file would be placed. It is the configuration file to train.\

Some points to be noted during use of this file are listed below :
* Open yolo-obj.cfg file and search for **classes** and put the desired number of your class there
* Search for **num**.  By default in *yolov3* it is given **9** but it is a value which helps to determine to get the filter numbers.  Formula of filter number is `filters= (num/3)∗(classes+ 5)`
*  To put the value of filters just search the word **yolo**, you will see that this word appeared **3** times and before this there is a word **filters**.  Change the value of it.  Some other filters words could be appeared but no need to change them. You can check [this issue](https://github.com/pjreddie/darknet/issues/236) for this point.
\
In darknet/ folder put darknet53.conv.74 file.  It is a configured training file.  You can also download itfrom [here](https://github.com/mathieuorhan/darknet).

# Training & Testing method
* To use GPU open the **Makefile**and change **GPU = 0 to GPU = 1** and **make** the repository again.
* To  initiate  training  process  go  to darknet directory  and  open  a  terminal  then  write  the  following command and hit Enter `./darknet detector train data/obj.data cfg/yolo-obj.cfg darknet53.conv.743*`. 
* Already trained weights also could be used as a pre trained weight file.  For example, trained weights name  is *xyzweights.weights* then as like as former point in  the  terminal  type `./darknet  detector  traindata/obj.data cfg/yolo-obj.cfg backup/xyzweights`.
* To test the image with trained weight and save the output write the following command from the darknet directory `./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj1200.weightsdata/testimage.jpg -out testoutput`.
* To test bunch of images or video using trained YOLO weight file you can follow [this](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/yolo_testing.ipynb) and [this](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/yolo_testing_video.py).

# Classification using Keras

This  process  can  identify  object  with  proper  classification  where  input  image  will  contain  only  the object except the surroundings which can be addressed as also as cropped image.  These cropped image is generated with [this code](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/capture_image_UE4.py).  To do the process training and testing file could be listed in a csv file which can be done using [this code](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/classification_UE4_image/creation_of_csv_file_for_training.py).  To initiate training [this code](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/classification_UE4_image/csv_file_training_for_classification.py) can be used.  During training one can do image augmentation using built-in library, details would be found here [here](https://keras.io/preprocessing/image/).
