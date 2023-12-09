# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import pprint
import os
import random
import re
import nltk
import cv2
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm
from random import shuffle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


def import_images(folder_name):
    files = [os.path.join(folder_name, file) for file in os.listdir(folder_name) if file.endswith('.png')]
    return files

def get_random_train_and_test_files(train_sample_size, exists_images, not_exists_images):
    all_files = exists_images + not_exists_images
    training_files = random.sample(all_files, int(len(all_files)*train_sample_size))
    testing_files= [file for file in all_files if file not in training_files]
    return training_files, testing_files

def get_image_data_with_labels(files, exists_folder):
    images_with_labels = []
    for file in files:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        matrix = np.array(image)
        label = np.array([1,0]) if file.startswith(exists_folder + '/') else np.array([0,1])
        images_with_labels.append([matrix, label])
    return images_with_labels

def train_model(training_data):
    training_matrix = np.array([i[0] for i in training_data]).reshape(-1,64,64,1)
    training_labels = np.array([i[1] for i in training_data])
    model = Sequential()

    model.add(InputLayer(input_shape = [64,64,1]))
    model.add(Conv2D(filters=32, kernel_size=5,strides=1,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=5,padding='same'))

    model.add(Conv2D(filters=50,kernel_size=5,strides=5,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=5,padding='same'))

    model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=5,padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2,activation='softmax'))
    optimizer=Adam(lr=1e-3)

    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=training_matrix,y=training_labels,epochs=50,batch_size=100)
    model.summary()
    return model

def get_predictions(model, testing_data):
    testing_matrix = np.array([i[0] for i in testing_data]).reshape(-1,64,64,1)
    predictions = model.predict(testing_matrix)
    return predictions

def show_predictions(predictions, testing_data, exists_label, not_exists_label):
    fig = plt.figure(figsize=(14,14))
    for index, prediction in enumerate(predictions):
        y = fig.add_subplot(6,5,index+1)
        
        if np.argmax(prediction) == 1:
            str_label=not_exists_label
        else:
            str_label=exists_label
            
        y.imshow(testing_data[index][0], cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.show(block=True)

def compute_confusion_matrix(testing_data, predictions):
    return 'confusion_matrix'


def main():
    # INITIALIZE INPUTS
    exists_folder = 'bread'
    not_exists_folder = 'notbread'
    train_sample_size = 0.85

    # IMPORTING THE IMAGES
    exists_images = import_images(exists_folder)
    not_exists_images = import_images(not_exists_folder)

    # GETTING THE TEST AND TRAINING DATASETS
    training_files, testing_files = get_random_train_and_test_files(train_sample_size, exists_images, not_exists_images)

    # GETTING THE IMAGE DATA WITH LABELS
    training_data = get_image_data_with_labels(training_files, exists_folder)
    testing_data = get_image_data_with_labels(testing_files, exists_folder)

    # TRAINING THE MODEL
    model = train_model(training_data)

    # CALCULATE PREDICTIONS
    predictions = get_predictions(model, testing_data)

    # SHOW PREDICTIONS
    show_predictions(predictions, testing_data, exists_folder, not_exists_folder)

    # COMPUTE CONFUSION MATRIX
    confusion_matrix = compute_confusion_matrix(testing_data, predictions)

    print(confusion_matrix)


main()