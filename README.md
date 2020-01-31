# Autonomous Image Perception System for Inventory Robot
The goal of this project is to develop a system which will be used in a retailer shop to count and filling objects from the shell. The project is conducting by [Institute for Artificial Intelligence](https://ai.uni-bremen.de/). Ideas and developed steps are discussed below.

# Requirements
Unreal Engine (Tested in Windows 10, 64 bit machine with v4.16, 4.19)
UnrealCV(fast branch)
Python 3.6
Windows 10, 64 bit


# Creation of image dataset from Unreal Engine
The goal of this part is to develop a system which will be capable of capturing image of object which isimported as an actor in UE4.  A brief introduction about instaling UE4, UnrealCV and basics of the codeis available [here](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/unreal_cv_documentaion.pdf).

Full code is available [here](https://github.com/atifkarim/unreal_cv_image_manipulation/tree/master/capture_UE4_image).
Step by step the working method of the automated image capturing tool is described below.

* First of all of the objects which will be captured by UE4 will be placed or imported in the game
* A [configuration file](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/config_file_capture_image.json) is made where all of the angles and objects name with the radius(distance between camera and the object) are mentioned.
* Before running the [main python file](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/capture_image_UE4.py) it is obvious to turn on thePLAY moodin UE4
* In  [main.py](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_UE4_image/capture_image_UE4.py) two  functions  are  called,  one  for  cropping  the  object  from  the  RGB  image∗using  themask information and another is for doing the annotation
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
  
  



![GitHub Logo](https://www.pexels.com/photo/beautiful-beauty-blue-bright-414612/)
  
  
