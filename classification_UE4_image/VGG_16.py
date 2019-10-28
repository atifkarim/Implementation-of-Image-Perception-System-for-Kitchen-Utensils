import numpy as np
import matplotlib.pyplot as plt
# from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import models
from keras import layers
from keras import optimizers

import glob, os
from skimage import io, color, exposure, transform
from skimage.color import rgb2gray

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Flatten, Dense

# train_dir = '/home/atif/machine_learning_stuff/ml_image/copy_image/train'
# validation_dir = '/home/atif/machine_learning_stuff/ml_image/copy_image/validation'

train_dir = '/home/atif/machine_learning_stuff/ml_image/train_image_AI'
validation_dir = '/home/atif/machine_learning_stuff/ml_image/validation_image_AI'
path_pre_trained_model = '/home/atif/machine_learning_stuff/model_file_keras/'
IMG_SIZE = 100
IMG_depth = 3 # for RGB 3, for B&W it will be 1
NUM_CLASSES = 19


# # All layers are freezed. Same as pre trained weights

from keras.applications import VGG16

# Download the VGG 16 model . Check here https://github.com/keras-team/keras/issues/5257#issuecomment-314628173 .
# with TF backend maybe not possible to use VGG 16 with channel first
# vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_depth,IMG_SIZE,IMG_SIZE))

#Load the VGG model
loaded_vgg = VGG16(weights=path_pre_trained_model+'vgg_top_48_shape/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE, IMG_depth))


# Freeze all the layers
for layer in loaded_vgg.layers[:-4]:
    layer.trainable = False
    
# If I write here layer.trainable = True then layer from top will be true. WHY?? because the indexing of the array is taking this decision. fro  top
# to before the last 4 layer I have told to be False rest are will be true


# Check the trainable status of the individual layers
for layer in loaded_vgg.layers:
    print(layer, layer.trainable)


# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(loaded_vgg)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
#model.add(layers.Dense(2048, activation='relu'))
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()


def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,centre[1]-min_side//2:centre[1]+min_side//2,:]
#    img = rgb2gray(img)

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
#     img = np.rollaxis(img,-1) # this line is doing the channel fisrt operation

    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])
#     return str(img_path.split('/')[-2]) # returning the folder name. If use -1 that means image name. consider the img_path.


imgs = []
labels = []
root_dir = '/home/atif/machine_learning_stuff/ml_image/train_image_AI/'
#path='/home/atif/training_by_several_learning_process/flower_photos/00000/'

#all_img_paths = glob.glob(path+ '5547758_eea9edfd54_n_000.jpg')

all_img_paths = glob.glob(os.path.join(root_dir, '*/*')) #I have done the training with .png format image. If another type of image will come 
                                                                                    #them .png will be changed by that extension
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    try:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

        if len(imgs)%1200 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            #print("get it 2")
    except (IOError, OSError):
        print('missed', img_path)
        pass


X = np.array(imgs, dtype='float32') #Keeping the image as an array
X = X.reshape(len(imgs),IMG_SIZE,IMG_SIZE,IMG_depth) # write (IMG_SIZE,IMG_SIZE,1 if you want channel last; 1= grayscale;3=RGB)
# Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
Y = keras.utils.to_categorical(labels, NUM_CLASSES)

print('X shape: ', X.shape,' type: ',type(X))
print('Y shape: ', Y.shape,' type: ',type(Y))


lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


path = '/home/atif/machine_learning_stuff/model_file_keras/'


import datetime
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("current time:", current_time)


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

batch_size = 32
epochs = 2
do_train_model=model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,verbose=2,
          #np.resize(img, (-1, <image shape>)
          callbacks=[LearningRateScheduler(lr_schedule) ,ModelCheckpoint(path+str(current_time)+'_VGG_16_image_size_100_epoch_'+str(epochs)+'.h5', save_best_only=True)])


#model.save(path+str(current_time)+'_vgg16_epoch_'+str(epochs)+'.h5')
