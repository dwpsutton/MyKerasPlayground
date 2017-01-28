

import numpy as np, matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, Flatten, Dense, Dropout, MaxPooling2D, Activation

DTYPE= 'float32'
epochs= 10
batch_size= 32
fout='cifar10_10epochs.hd5'

(x_train, y_train), (x_test, y_test)= cifar10.load_data()

# Pre-process the data into depth 1 images.  The size is 32*32 pixels each

print 'Found ' + str(x_train.shape[0]) + ' training images'
X_train= x_train.reshape( x_train.shape[0],3,32,32 ).astype(DTYPE) / 255
X_test= x_test.reshape( x_test.shape[0],3,32,32 ).astype(DTYPE) / 255

# Cool function that one hot encodes the categorical label field
Y_train= keras.utils.np_utils.to_categorical(y_train,10)
Y_test= keras.utils.np_utils.to_categorical(y_test,10)

# Define a VGG conv-net, 8 layers
network= keras.models.Sequential([
                                  Convolution2D(32,3,3,activation='relu',input_shape= (3,32,32),dim_ordering='th' ),
                                  Convolution2D(32,3,3,activation='relu',dim_ordering='th' ),
                                  MaxPooling2D(pool_size=(2,2)),
                                  Dropout(0.5),
                                  Convolution2D(64,3,3,activation='relu',dim_ordering='th' ),
                                  Convolution2D(64,3,3,activation='relu',dim_ordering='th' ),
                                  MaxPooling2D(pool_size=(2,2)),
                                  Dropout(0.5),
                                  Flatten(),
                                  Dense(128,activation='relu'),
                                  Dropout(0.5),
                                  Dense(10,activation='softmax')
                                  ])

network.compile(optimizer= 'adam',
                loss= 'categorical_crossentropy',
                metrics=['accuracy']
                )

datagen= keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                                                      samplewise_center=False,
                                                      featurewise_std_normalization=True,
                                                      samplewise_std_normalization=False,
                                                      zca_whitening=False,
                                                      rotation_range=0.,
                                                      width_shift_range=0.,
                                                      height_shift_range=0.,
                                                      shear_range=0.,
                                                      zoom_range=0.,
                                                      channel_shift_range=0.,
                                                      fill_mode='nearest',
                                                      cval=0.,
                                                      horizontal_flip=False,
                                                      vertical_flip=False,
                                                      rescale=None,
                                                      dim_ordering='th')
datagen.fit(X_train)


network.fit_generator(datagen.flow( X_train,Y_train, batch_size=batch_size ), samples_per_epoch= len(Y_train),nb_epoch=epochs,verbose=1)

print 'Found ' + str(X_test.shape[0]) + ' test images'

score = network.evaluate_generator(datagen.flow( X_test,Y_test,batch_size=32),len(Y_test))

print 'TEST SET: loss= '+str(score[0])+', accuracy= '+str(score[1])


#network.predict_proba(XX)

# Save the model for later
network.save(fout)
