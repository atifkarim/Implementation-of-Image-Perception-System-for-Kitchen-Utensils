# This file contains information to train and test model using YOLO

For a fresh startup use [this link](https://github.com/atifkarim/unreal_cv_image_manipulation#object-detetion-and-localization-using-yolo) from where a quick setup, essential link, codes and instructions of arranging folder and images you will get. In the following lines step by step
I will try to arrange the whole process to initiate a raining and tsting using **YOLO**.

* Install and make **darknet** in your system by following [this](https://pjreddie.com/darknet/install/)
* Clone [this repo](https://github.com/ManivannanMurugavel/YOLO-Annotation-Tool) which will be used to create file for YOLO format
  * some modification will be happened here.
  * First of all in the **Images** folder put all of the RGB images folders of yours which you have used to generate annotation (class with 
  region of interest information) file. Regarding this creation you will get information from [here](https://github.com/atifkarim/unreal_cv_image_manipulation#creation-of-image-dataset-from-unreal-engine)
  * Then go the **Labels** folder and delete all except **out** folder. If this **out** folder is not there ust create it. If it was there
  already just make it empty. In **Labels** folder put all the annotated text file with their folders.
  * Now delete the **convert.py** file and put [this file](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/convert_yolo_dataset_modified.py) there.
  * Run [this file](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/convert_yolo_dataset_modified.py) file and 
  it will create the converted YOLO format text files and stores them in **/Labels/out/** directory. Just pay attention to some 
  points before running the code --
    * Set the [directory of Images](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/convert_yolo_dataset_modified.py#L10) properly
    * Set the path of [current file](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/convert_yolo_dataset_modified.py#L77) and
    [new file](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/convert_yolo_dataset_modified.py#L78) properly
    properly. **MAY BE DUE TO SOME CHANGES IN CODE THIS LINE COULD BE CHANGED AND THEN THIS HYPERLINK WILL NOT NAVIGAT YOU TO
    THE CORRECT LINE. SORRY FOR THIS BUT YOU WILL FIND IT EASILY**
    * Set up your [image type](https://github.com/atifkarim/unreal_cv_image_manipulation/blob/master/testing_YOLO/convert_yolo_dataset_modified.py#L156)
    
