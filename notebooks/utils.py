from keras.utils import Sequence
from sklearn.utils import shuffle
import numpy as np
import csv
import cv2

import os


def read_samples(data_dir):
    samples = []
    train_runs = [os.path.join(data_dir, run_dir) for run_dir in ['run1', 'run2']]
    for run_dir in train_runs:
        driving_log = os.path.join(run_dir, 'driving_log.csv')
        with open(driving_log, 'r') as _file:
            log = csv.reader(_file)
            for line in log:
                samples.append(line)
    print('Loaded {} samples.'.format(len(samples)))
    return samples


class ImageGenerator(Sequence):
    def __init__(self, samples, batch_size, gray=True, flip=False):
        self.train_images = []
        self.steering_angles = []
        for sample in samples:
            self.train_images.extend([(False, img) for img in sample[: 3]])
            self.steering_angles.extend([float(sample[3])] * 3)
        self.steering_angles = np.array(self.steering_angles)
        if flip:
            for sample in samples:
                self.train_images.extend([(True, img) for img in sample[: 3]])
            self.steering_angles = np.hstack((self.steering_angles, -self.steering_angles))
        self._len = len(self.steering_angles)
        self.batch_size = batch_size
        if gray:
            self.cv_flag = cv2.IMREAD_GRAYSCALE
        else:
            self.cv_flag = cv2.IMREAD_COLOR

    def __len__(self):
        return (self._len + self.batch_size - 1) // self.batch_size

    def __getitem__(self, item):
        x_batch = []
        for to_flip, img_path in self.train_images[item: item + self.batch_size]:
            img = cv2.imread(img_path, flags=self.cv_flag)
            if to_flip:
                img = cv2.flip(img, 1)
            img = img[60: 140]
            x_batch.append(img)
        x_batch = np.array(x_batch)
        y_batch = self.steering_angles[item: item + self.batch_size]
        if self.cv_flag == cv2.IMREAD_GRAYSCALE:
            x_batch = np.expand_dims(x_batch, axis=3)
        return x_batch, y_batch

    def on_epoch_end(self):
        self.train_images, self.steering_angles = shuffle(self.train_images, self.steering_angles)


class CenterImageGenerator(Sequence):
    def __init__(self, samples, batch_size, gray=True, flip=False, correction=0):
        self.train_images = []
        self.steering_angles = []
        for sample in samples:
            self.train_images.append((False, sample[0]))
            self.steering_angles.append(float(sample[3]) + correction)
        self.steering_angles = np.array(self.steering_angles)
        if flip:
            for sample in samples:
                self.train_images.append((True, sample[0]))
            self.steering_angles = np.hstack((self.steering_angles, -self.steering_angles))
        self._len = len(self.steering_angles)
        self.batch_size = batch_size
        if gray:
            self.cv_flag = cv2.IMREAD_GRAYSCALE
        else:
            self.cv_flag = cv2.IMREAD_COLOR

    def __len__(self):
        return (self._len + self.batch_size - 1) // self.batch_size

    def __getitem__(self, item):
        x_batch = []
        for to_flip, img_path in self.train_images[item: item + self.batch_size]:
            img = cv2.imread(img_path, flags=self.cv_flag)
            if to_flip:
                img = cv2.flip(img, 1)
            img = img[60: 140]
            x_batch.append(img)
        x_batch = np.array(x_batch)
        y_batch = self.steering_angles[item: item + self.batch_size]
        if self.cv_flag == cv2.IMREAD_GRAYSCALE:
            x_batch = np.expand_dims(x_batch, axis=3)
        return x_batch, y_batch

    def on_epoch_end(self):
        self.train_images, self.steering_angles = shuffle(self.train_images, self.steering_angles)
