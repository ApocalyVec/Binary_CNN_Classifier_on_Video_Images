from datetime import date, datetime
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.models import load_model
from keras.models import save_model

from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow
import cv2
import numpy as np

# file related imports
import os, os.path
import shutil
import random

on_frame_path = 'thumb_on_frames'
not_frame_path = 'thumb_not_frames'

train_folder_name = 'training_set'
test_folder_name = 'test_set'

on_category_name = 'on'
not_category_name = 'not'

trained_model_folder_name = 'trained_model'

def ThumouseFrameCapture(path: str, category:str):
    global not_frame_path, on_frame_path

    vidcap = cv2.VideoCapture(path)
    # convert the video to series of images
    count = 0
    success = 1
    print('Processing frames: file = ' + path + ', Category = ' + category)

    if category == 'on':
        while success:
            success, img = vidcap.read()
            cv2.imwrite(os.path.join(on_frame_path, (path + "%d.jpg" % count)), img)  # save to ON frame path

            print('     created ' + path + "%d.jpg" % count)
            if cv2.waitKey(10) == 27:
                break
            count += 1
    elif category == 'not':
        while success:
            success, img = vidcap.read()
            cv2.imwrite(os.path.join(not_frame_path, (path + '_' + "%d.jpg" % count)), img)  # save to NOT frame path

            print('     created ' + path + "%d.jpg" % count)
            if cv2.waitKey(10) == 27:
                break
            count += 1
    else:
        raise('Invalid Category ' + category, 'Category must be either On or Not')

    print('     Validating frame images')
    for fn in os.listdir(on_frame_path):
        full_fn = os.path.join(on_frame_path, fn)
        try:
            Image.open(full_fn)
        except OSError:
            os.remove(full_fn)
            print('         ' + fn + ' is Invalid, removed')

    for fn in os.listdir(not_frame_path):
        full_fn = os.path.join(not_frame_path, fn)
        try:
            Image.open(full_fn)
        except OSError:
            os.remove(full_fn)
            print(fn + ' is Invalid, removed')


def convertFrames(on_video_list, not_video_list):
    global not_frame_path, on_frame_path

    # create frame folders
    os.mkdir(on_frame_path)
    os.mkdir(not_frame_path)

    for on_v in on_video_list:
        ThumouseFrameCapture(on_v, 'on')
    for not_v in not_video_list:
        ThumouseFrameCapture(not_v, 'not')

    # ThumouseFrameCapture('thumb_not_on_pointing.mp4', False)
    # ThumouseFrameCapture('thumb_not_on_pointing_zehua.mp4', False)
    #
    # # on videos and their frames
    # ThumouseFrameCapture('thumb_on_pointing.mp4', True)
    # ThumouseFrameCapture('thumb_on_pointing_zehua.mp4', True)


def separate_train_test(path: str, category: str, test_percentage: float):
    global train_folder_name, test_folder_name

    if not os.path.isdir(train_folder_name):  # check if the training set folder exists
        os.mkdir(train_folder_name)
    if not os.path.isdir(test_folder_name):  # check if the test set folder exists
        os.mkdir(test_folder_name)

    file_num = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    test_num = int(file_num * test_percentage)

    # create the train and test folder
    train_path = os.path.join(train_folder_name, category)
    test_path = os.path.join(test_folder_name, category)
    os.mkdir(train_path)
    os.mkdir(test_path)

    count = 1
    test_fn_list = []

    # copy the test files
    print('Total file number: ' + str(file_num))
    print('transferring test files')
    while count <= test_num:
        target_test_fn = random.choice(os.listdir(path))
        test_full_fn = os.path.join(path, target_test_fn)
        if os.path.isfile(test_full_fn) and target_test_fn not in test_fn_list:  # making sure that the files are unique
            test_fn_list.append(test_full_fn)
            shutil.copy(test_full_fn, test_path)
            print('     ' + 'Copying ' + str(count) + ' of ' + str(test_num) + ' ' + test_full_fn + ' to ' + test_path)
            count += 1

    origin = os.listdir(path)
    # copy the train files
    print('transferring train files')
    for remaining_fn in origin:
        train_full_fn = os.path.join(path, remaining_fn)
        if os.path.isfile(train_full_fn) and train_full_fn not in test_fn_list:
            shutil.copy(train_full_fn, train_path)
            print('     Copied ' + train_full_fn + ' to ' + train_path)


def main(isConvertFrames: bool, isCreateTrainTest: bool, on_videos=[], not_videos=[]):
    global on_category_name, not_category_name

    if isConvertFrames:
        print('Removing Old Frames dir')
        if os.path.isdir(on_frame_path):
            shutil.rmtree(on_frame_path)
        if os.path.isdir(not_frame_path):
            shutil.rmtree(not_frame_path)
        convertFrames(on_videos, not_videos)  # use this line to create frames as jpeg

    if isCreateTrainTest:
        #  remove the test and training set if such exist
        if os.path.isdir(test_folder_name):
            print('Removing old test_set')
            shutil.rmtree(test_folder_name)
        if os.path.isdir(train_folder_name):
            print('Removing old training_set')
            shutil.rmtree(train_folder_name)

        # create training set and test set
        separate_train_test(on_frame_path, on_category_name, 0.2)
        separate_train_test(not_frame_path, not_category_name, 0.2)

    # send training and test set to keras
    train_datagen = ImageDataGenerator(horizontal_flip=True, rescale=1. / 255)  # flip the image for training
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_folder_name,
                                                     target_size=(128, 128),
                                                     batch_size=32,
                                                     class_mode='binary')
    test_set = test_datagen.flow_from_directory(test_folder_name,
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')
    # create the CNN
    classifier = Sequential()
    classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 3),
                          activation='relu'))  # using 3x3 pixels as a conv window
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # using 3x3 pixels as a conv window
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # using 3x3 pixels as a conv window
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Flatten())
    classifier.add(Dense(units=300, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    classifier.fit_generator(training_set,
                             steps_per_epoch=len(training_set.filenames),
                             epochs=5,
                             validation_data=test_set,
                             validation_steps=len(test_set.filenames))

    if not os.path.isdir(trained_model_folder_name):
        os.mkdir(trained_model_folder_name)
    classifier.save(os.path.join(trained_model_folder_name, 'model_' + str(datetime.today())))

if __name__ == '__main__':
    on_videos = ['thumb_on_pointing.mp4', 'thumb_on_pointing_zehua.mp4']
    not_videos = ['thumb_not_on_pointing.mp4', 'thumb_not_on_pointing_zehua.mp4']
    main(isConvertFrames=True, isCreateTrainTest=True, on_videos=on_videos, not_videos=not_videos)
