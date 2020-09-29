import tensorflow as tf
import keras
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils


#Loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Normalizing data
x_train_std=np.std(x_train,axis=(0,1,2))
x_train_mean=np.mean(x_train,axis=(0,1,2))
X_train=(x_train-x_train_mean)/x_train_std

x_test_std=np.std(x_test,axis=(0,1,2))
x_test_mean=np.mean(x_test,axis=(0,1,2))
X_test=(x_test-x_test_mean)/x_test_std

number_classes=np.max(y_train)+1

#Reshaping to include a grayscale channel dimension. For RGB, one would have a 3 there autormatically
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)


#Making data small so we can test the nn
#X_train=X_train[0:10,:,:,:]
#X_test=X_test[0:10,:,:,:]
#y_train=y_train[0:10]
#y_test=y_test[0:10]

#to categorical ???
y_train = np_utils.to_categorical(y_train,number_classes)
y_test = np_utils.to_categorical(y_test,number_classes)

#learning schedule
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

#creating model
model=Sequential()
#coefficient keeping network simple
weight_decay = 1e-4

#First layer - Convolutional, Normalization, pooling, and dropout
model.add(Conv2D(16, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#Second layer - Convolutional, Normalization, pooling, and dropout. The filter is now bigger
model.add(Conv2D(32, (3,3), padding='same',  activation="relu", kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#last layer is flatten
model.add(Flatten())
model.add(Dense(number_classes, activation='softmax'))
 
model.summary()


#Augmenting data
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(X_train)

#training
batch_size = 64


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.001,decay=1e-6), metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=X_train.shape[0] // batch_size,epochs=100,\
                    verbose=1,validation_data=(X_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5') 
 
#testing
scores = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))