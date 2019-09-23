#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from skimage import io, color, exposure, transform
from skimage.color import rgb2gray
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split  #it came from update scikit learn. https://stackoverflow.com/questions/40704484/importerror-no-module-named-model-selection
import os
import glob
import h5py
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.pooling import MaxPooling2D

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')

from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams

#import keras

NUM_CLASSES = 16 # change it with respect to the desired class
IMG_SIZE = 48 # change it if it desired
IMG_depth = 3 # for RGB 3, for B&W it will be 1


# In[35]:


import datetime
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("current time:", current_time)


# In[9]:


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
    img = np.rollaxis(img,-1)

    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])
#     return str(img_path.split('/')[-2]) # returning the folder name. If use -1 that means image name. consider the img_path.


# In[10]:


imgs = []
labels = []
path = '/home/atif/machine_learning_stuff/model_file_keras/'
root_dir = '/home/atif/machine_learning_stuff/ml_image/train_image_AI/'
#path='/home/atif/training_by_several_learning_process/flower_photos/00000/'

#all_img_paths = glob.glob(path+ '5547758_eea9edfd54_n_000.jpg')

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.png')) #I have done the training with .png format image. If another type of image will come 
                                                                                    #them .png will be changed by that extension
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    try:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

        if len(imgs)%1248 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            #print("get it 2")
    except (IOError, OSError):
        print('missed', img_path)
        pass


# In[12]:


X = np.array(imgs, dtype='float32') #Keeping the image as an array
X = X.reshape(len(imgs),IMG_depth,IMG_SIZE,IMG_SIZE) # write (IMG_SIZE,IMG_SIZE,1 if you want channel last; 1= grayscale;3=RGB)
# Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
Y = keras.utils.to_categorical(labels, NUM_CLASSES)

print('X shape: ', X.shape,' type: ',type(X))
print('Y shape: ', Y.shape,' type: ',type(Y))


# In[24]:


# conv_base_1 = VGG19(weights='imagenet', include_top=True)


# In[37]:


# conv_base_1.summary()


# In[19]:


from keras.applications import VGG19
# conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(3,48,48))

pre_trained_weight = VGG19(weights=path+'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False, input_shape=(IMG_depth,IMG_SIZE,IMG_SIZE))


# In[11]:


# # if want to download weight file

# from keras.applications import InceptionResNetV2

# conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(48,48,3))


# In[ ]:


# from keras.applications import InceptionResNetV2

# conv_base = InceptionResNetV2(weights='/home/atif/machine_learning_stuff/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                               include_top=False, input_shape=(3,150,150))


# In[20]:


for layer in pre_trained_weight.layers[:-4]:
    layer.trainable = False
# Check the trainable status of the individual layers
for layer in pre_trained_weight.layers:
    print(layer, layer.trainable)


# In[26]:


# pre_trained_weight.summary()


# In[28]:


def cnn_model():
    model = Sequential()
    
    model.add(pre_trained_weight)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


# In[29]:


model = cnn_model()


# In[30]:


model.summary()


# In[31]:


# print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
# conv_base.trainable = False
# print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))


# In[32]:


lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])


# In[33]:


path = '/home/atif/machine_learning_stuff/model_file_keras/'


# In[36]:


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

batch_size = 32
epochs = 50
do_train_model=model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,verbose=2,
          #np.resize(img, (-1, <image shape>)
          callbacks=[LearningRateScheduler(lr_schedule),ModelCheckpoint(path+str(current_time)+'_vgg19_epoch_'+str(epochs)+'.h5', save_best_only=True)])


# In[ ]:




