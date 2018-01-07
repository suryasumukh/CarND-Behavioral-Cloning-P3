from keras.utils import Sequence
from sklearn.utils import shuffle
import numpy as np
import csv
import cv2

import os


def read_samples(data_dir):
    samples = []
    run_dirs = ['sample-data']
    train_runs = [os.path.join(data_dir, run_dir) for run_dir in run_dirs]
    for run_dir in train_runs:
        driving_log = os.path.join(run_dir, 'driving_log.csv')
        with open(driving_log, 'r') as _file:
            log = csv.reader(_file)
            for line in log:
                samples.append(line)
    print('Loaded {} samples.'.format(len(samples)))
    return samples


class ImageGenerator(Sequence):
    def __init__(self, samples, batch_size, corr=0.0):
        self.batch_size = batch_size
        self.corr = corr

        self.samples = []
        self.steering_angles = []

        for sample in samples:
            for img_path in sample[: 3]:
                self.samples.append((False, img_path))
                self.samples.append((True, img_path))
            angle = float(sample[3])
            angles = [angle, -angle, angle+corr, -(angle+corr), angle-corr, -(angle-corr)]
            self.steering_angles.extend(angles)

        self.steering_angles = np.array(self.steering_angles)
        self._len = len(self.steering_angles)

        self.samples, self.steering_angles = shuffle(self.samples, self.steering_angles)

    def __len__(self):
        return (self._len + self.batch_size - 1) // self.batch_size

    def preprocess(self, sample):
        flip, img_path = sample
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[70: 140, :, 2]
        if flip:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, (320, 160))
        return np.expand_dims(image, axis=2)

    def __getitem__(self, item):
        x_batch = np.empty([self.batch_size, 160, 320, 1])
        y_batch = []
        i = 0
        while len(y_batch) < self.batch_size:
            idx = np.random.choice(self._len)
<<<<<<< HEAD
            if np.absolute(self.steering_angles[idx]) <= 0.2:
=======
            if 1.5 < np.absolute(self.steering_angles[idx]) < 0.3:
                if np.random.uniform() < 0.3:
                    y_batch.append(self.steering_angles[idx])
                    x_batch[i] = self.preprocess(self.samples[idx])
                    i += 1
            elif 0 < np.absolute(self.steering_angles[idx]) < 1.0:
>>>>>>> 548aa04aeb5e4694e91b5a8251ee80593b80003f
                if np.random.uniform() < 0.3:
                    y_batch.append(self.steering_angles[idx])
                    x_batch[i] = self.preprocess(self.samples[idx])
                    i += 1
            else:
                y_batch.append(self.steering_angles[idx])
                x_batch[i] = self.preprocess(self.samples[idx])
                i += 1

        return x_batch, np.array(y_batch)

    def on_epoch_end(self):
        self.samples, self.steering_angles = shuffle(self.samples, self.steering_angles)


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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
            if to_flip:
                img = cv2.flip(img, 1)
            img = img[60: 140]
            img = cv2.resize(img, (320, 160), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=2)
            x_batch.append(img)
        x_batch = np.array(x_batch)
        y_batch = self.steering_angles[item: item + self.batch_size]
        if self.cv_flag == cv2.IMREAD_GRAYSCALE:
            x_batch = np.expand_dims(x_batch, axis=3)
        return x_batch, y_batch

    def on_epoch_end(self):
        self.train_images, self.steering_angles = shuffle(self.train_images, self.steering_angles)
