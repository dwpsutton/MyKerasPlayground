

import numpy as np, matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers import Convolution2D, Flatten, Dense, Dropout, MaxPooling2D, Activation

DTYPE= 'float32'
epochs= 10
batch_size= 32
fout='mnist.hd5'

(x_train, y_train), (x_test, y_test)= mnist.load_data()

# Pre-process the data into depth 1 images.  The size is 28*28 pixels each

print 'Found ' + str(x_train.shape[0]) + ' training images'
X_train= x_train.reshape( x_train.shape[0],1,28,28 ).astype(DTYPE) / 255
X_test= x_test.reshape( x_test.shape[0],1,28,28 ).astype(DTYPE) / 255

# Cool function that one hot encodes the categorical label field
Y_train= keras.utils.np_utils.to_categorical(y_train,10)
Y_test= keras.utils.np_utils.to_categorical(y_test,10)

# Define a VGG conv-net, 8 layers
network= keras.models.Sequential([
                                  Convolution2D(32,3,3,activation='relu',input_shape= (1,28,28),dim_ordering='th' ),
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

network.fit(X_train,Y_train, batch_size=batch_size, nb_epoch=epochs,verbose=1)

print 'Found ' + str(X_test.shape[0]) + ' test images'
probs= network.predict_proba(X_test,batch_size=batch_size)

score = network.evaluate(X_test,Y_test,verbose=0)

print 'TEST SET: loss= '+str(score[0])+', accuracy= '+str(score[1])

# Save the model for later
network.save(fout)
