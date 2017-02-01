

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.regularizers import l2
from keras import optimizers
import numpy as np, glob, h5py
from datetime import datetime,timedelta
#import tensorflow as tf

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/Users/davidsutton/Downloads/kaggle/train'
validation_data_dir = '/Users/davidsutton/Downloads/kaggle/val'
nb_train_samples = 21999
nb_validation_samples = 3001
nb_epoch = 1
l2_lam= 0.2

weights_fname= '/Users/davidsutton/Downloads/kaggle/vgg16_weights.h5'

#with tf.device('/gpu:0'):

def load_VGG16(fname):
    
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    f = h5py.File(fname)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print 'Model loaded.'
    return model


def create_bottleneck_features(model,train_data_dir,validation_data_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
                                            train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=32,
                                            classes= ['dogs','cats'],
                                            class_mode=None,
                                            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('/Users/davidsutton/Downloads/kaggle/bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    
    generator = datagen.flow_from_directory(
                                            validation_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=32,
                                            classes= ['dogs','cats'],
                                            class_mode=None,
                                            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator,nb_validation_samples)
    np.save(open('/Users/davidsutton/Downloads/kaggle/bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def build_top_model():
    
    # My little network
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape= train_data.shape[1:])) #(3, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256,W_regularizer=l2(l2_lam)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256,W_regularizer=l2(l2_lam)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
    
def fit_top_model(model):
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
        
    print 'fitting top level network...'
    model.fit(train_data,train_labels,
              nb_epoch=nb_epoch,batch_size=32,
              validation_data= (val_data, val_labels))
                  
    model.save_weights('weights_vgg1.hd5')
    return model




def fine_tune(vgg,top_model):
    vgg.add(top_model)
    for layer in vgg.layers[:25]:
        layer.trainable = False



    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    vgg.compile(loss='binary_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
                                       rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
                                                        train_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=32,
                                                        classes= ['dogs','cats'],
                                                        class_mode='binary')
        
    validation_generator = test_datagen.flow_from_directory(
                                                            validation_data_dir,
                                                            target_size=(img_height, img_width),
                                                            batch_size=32,
                                                            classes= ['dogs','cats'],
                                                            class_mode='binary')
                                                        
    # fine-tune the model
    vgg.fit_generator(
                      train_generator,
                      samples_per_epoch=nb_train_samples,
                      nb_epoch=nb_epoch,
                      validation_data=validation_generator,
                      nb_val_samples=nb_validation_samples)
                                                        
    vgg.save_weights('final_model.hd5')
    return vgg



if __name__ == '__main__':
    print 'loading vgg model: '+weights_fname
    vgg= load_VGG16(weights_fname)
    
    print 'creating bottleneck features'
    tc= datetime.now()
#    create_bottleneck_features(vgg,train_data_dir,validation_data_dir)

    print 'prepping data',(datetime.now() - tc).total_seconds()
    train_data= np.load(open('/Users/davidsutton/Downloads/kaggle/bottleneck_features_train.npy','r'))
    train_labels= np.concatenate( (np.zeros(len( glob.glob(train_data_dir+'/cats/*.jpg') )),
                                   np.zeros(len( glob.glob(train_data_dir+'/dogs/*.jpg') ))+1) )

    val_data= np.load(open('/Users/davidsutton/Downloads/kaggle/bottleneck_features_validation.npy','r'))
    val_labels = np.concatenate( (np.zeros(len( glob.glob(validation_data_dir+'/cats/*.jpg') )),
                                  np.zeros(len( glob.glob(validation_data_dir+'/dogs/*.jpg') ))+1) )

    print 'build and fit a top model'
    top_model= build_top_model()
#    top_model= fit_top_model(top_model)
    top_model.load_weights('weights_vgg1.hd5')

    print 'fine-tuning...'
    vgg= fine_tune(vgg,top_model)