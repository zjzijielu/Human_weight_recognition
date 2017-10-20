#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:37:09 2017

@author: luzijie
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.callbacks import History as history
from keras.models import model_from_json

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model_20b.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples = 3000
nb_validation_samples = 600
epochs = 10
batch_size = 20


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print("#####finished generator####")
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size, verbose=1)
    np.save('bottleneck_features_train_20batch.npy',
            bottleneck_features_train)
    print("#####finished bottleneck_feature_train####")
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size, verbose=1)
    print("#####finished bottleneck_feature_validation####")
    np.save('bottleneck_features_validation_20batch.npy',
            bottleneck_features_validation)


def train_top_model(dropout):
    train_data = np.load('bottleneck_features_train_20batch.npy')
    train_labels = np.array(
        [0] * int(nb_train_samples / 3) + [1] * int(nb_train_samples / 3) + 
        [2] * int(nb_train_samples / 3))

    train_labels = to_categorical(train_labels, num_classes=3)
    
    test_data = np.load('bottleneck_features_validation_20batch.npy')
    test_labels = np.array(
        [0] * int(nb_validation_samples / 3) + [1] * int(nb_validation_samples / 3) + 
        [2] * int(nb_validation_samples / 3))
    
    test_labels = to_categorical(test_labels, num_classes=3)
    
    #fully connected layer
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(3, activation='softmax'))
    print("####model finished####")

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print("####model compile finished####")

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test_data, test_labels))
    #scores = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print("####model fit finished####")
    # serialize model to JSON
    model_json = model.to_json()
    with open("human_weight_recognition_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(top_model_weights_path)
    return history 

dropout = 0.4

#save_bottlebeck_features()
history = train_top_model(dropout)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy\n batch size:' + str(batch_size) + " dropout: " + str(dropout))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()