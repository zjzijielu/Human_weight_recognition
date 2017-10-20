#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:52:47 2017

@author: luzijie
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import model_from_json
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.callbacks import History as history
from PIL import Image


Overweight = 0
Normal = 1
Skinny = 2

img_width, img_height = 150, 150
batch_size = 10
prediction_data_dir = 'predict'

# load json and create model
json_file = open('human_weight_recognition_model.json', 'r')
print("####json_file####")
loaded_model_json = json_file.read()
json_file.close()
print("####close file####")
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("bottleneck_fc_model_20b.h5")
print("Loaded model from disk")


datagen = ImageDataGenerator(rescale=1. / 255)

vgg_model = applications.VGG16(include_top=False, weights='imagenet')

generator = datagen.flow_from_directory(
    prediction_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

print("#####finished generator####")
#print(type(generator))
#print(generator.shape)

bottleneck_features_predict = vgg_model.predict_generator(
    generator, 1, verbose=1)


# classify the image
predictions = loaded_model.predict(bottleneck_features_predict)
print(predictions)
predictions = np.argmax(predictions, axis=1)
labels = []
for i in range(len(predictions)):
    if predictions[i] == 0:
        labels.append("Normal weight")
    elif predictions[i] == 1:
        labels.append("Overweight")
    elif predictions[i] == 2:
        labels.append("Underweight")
print(labels)

# display the predictions to our screen

path1 = 'predict/class1/'
path2 = 'predict/class2/'

paths = [path1, path2]

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        try:
            img = Image.open(path + image)
            loadedImages.append(img)
        except:
            pass

    return loadedImages   

counter = 0

plt.ion()

for i in paths:
    imgs = loadImages(i)
    for img in imgs:
        print(counter)
        plt.figure(counter)
        plt.title("picture" + str(counter)+"\nprediction: " + labels[counter])
        plt.imshow(img)
        plt.pause(0.0001)
            
        #print("prediction of " + str(counter)  + " : ", labels[counter])
        plt.show()
        counter+=1
        