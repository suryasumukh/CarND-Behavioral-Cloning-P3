from matplotlib import pyplot as plt

from glob import glob
import json
import csv
import os

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import LeakyReLU, ELU, Dropout

from keras.optimizers import Adam
from keras.layers import Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import read_samples
from utils import ImageGenerator
from utils import CenterImageGenerator

from sklearn.model_selection import train_test_split


DATA_DIR = os.path.join('data')


def append_path(line):
    line[0] = os.path.join(DATA_DIR, 'sample-data', line[0].strip())
    line[1] = os.path.join(DATA_DIR, 'sample-data', line[1].strip())
    line[2] = os.path.join(DATA_DIR, 'sample-data', line[2].strip())
    return line


def comma_model():
    # comma.ai model adaptation
    ch, row, col = 1, 70, 320  # camera format
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    model.add(Conv2D(8, (8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(256))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    adam = Adam(0.001)
    model.compile(optimizer=adam, loss="mse")
    return model


def main():
    # read data
    samples = read_samples(DATA_DIR)
    _ = samples.pop(0)
    samples = [append_path(line) for line in samples]

    # create train, validation and test sets
    train_set, test_set = train_test_split(samples, test_size=0.1)
    train_set, valid_set = train_test_split(train_set, test_size=0.1)

    # create data iterators to load and preprocess images
    train_iterator = ImageGenerator(train_set, batch_size=128, corr=0.2)
    valid_iterator = CenterImageGenerator(valid_set, batch_size=128)
    test_iterator = CenterImageGenerator(test_set, batch_size=128)

    model = comma_model()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
    verbose=1, mode='auto')
    filepath = 'comma.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
    save_best_only=True, save_weights_only=False, mode='auto', period=1)

    print('Training comma.ai model')
    model.fit_generator(generator=train_iterator, epochs=10,
                    validation_data=valid_iterator,
                    steps_per_epoch=len(train_iterator),
                    validation_steps=len(valid_iterator),
                    callbacks=[checkpoint, early_stopping]
                   )
    print('Done.')

    results = model.evaluate_generator(generator=test_iterator, steps=len(test_iterator))
    print('Test mse: {}'.format(results))

if __name__ == '__main__':
    main()
