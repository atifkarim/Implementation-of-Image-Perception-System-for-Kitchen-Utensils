# UnrealCV image_manipulation
The goal of this repo is to use Unreal Engine 4(UE4) for making dataset for Robotic perception system. For this purpose here I have introduced the combination of UE4 and UnrealCV.

# Requirements

Tested on Unreal Engine v4.16, 4.19

UnrealCV(fast branch)

Python 3.6

Windows 10, 64 bit


# code for capturing image as well as making dataset for YOLO and Keras

[Follow this](https://github.com/atifkarim/unreal_cv_image_manipulation/tree/master/capture_image_config_json)

Some pre requisite of executing 
[this code](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/capture_image_config_json/capture_image_UE4.py)

1/ [Here](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/d9c6e9af88067b8135d5ca100b13d9238dc6abba/capture_image_config_json/capture_image_UE4.py#L105) give the config file name that you want to use

2/ In the config file all minimal info has given which will control how to take the image like polar, azimuthal angle, viewmode of UE4, image type etc. Pay attention [here](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/d9c6e9af88067b8135d5ca100b13d9238dc6abba/capture_image_config_json/config_file_capture_image.json#L5), to give **minus**/**negative** sign before the step of polar angle.

3/ Befor running the code must press **PLAY BUTTON** of UE4

# code for capturing image and later use for detecting ROI, converting RGB image to black color except desired actor and detecting ROI from that image

By using this code you can capture LIT and MASK image and later you can use for detecting ROI of the desired actor.

1/ To capture the image with different mask color of actor follow [this](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/detect_ROI_for_multi_actor/detect_roi_while_every_actor_has_different_color_in_mask.ipynb). It will also save text file which contains the RGB info of every actor.

2/ To detect ROI and making RGB image balck follow [this](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/detect_ROI_for_multi_actor/detect_ROI_of_different_coloured_object.py)
