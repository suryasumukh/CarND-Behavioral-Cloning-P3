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

def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    return image,steering


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
        # image = cv2.resize(image, (320, 160))
        return np.expand_dims(image, axis=2)

    def __getitem__(self, item):
        x_batch = np.empty([self.batch_size, 70, 320, 1])
        y_batch = []
        i = 0
        while len(y_batch) < self.batch_size:
            idx = np.random.choice(self._len)
            angle = self.steering_angles[idx]
            if np.absolute(angle) < 0.1:
                if np.random.uniform() < 0.2:
                    y_batch.append(angle)
                    x_batch[i] = self.preprocess(self.samples[idx])
                    i += 1
            elif 0.15 < np.absolute(self.steering_angles[idx]) < 0.3:
                if np.random.uniform() < 0.2:
                    y_batch.append(angle)
                    x_batch[i] = self.preprocess(self.samples[idx])
                    i += 1
            else:
                y_batch.append(angle)
                x_batch[i] = self.preprocess(self.samples[idx])
                i += 1
        return x_batch, np.array(y_batch)

    def on_epoch_end(self):
        self.samples, self.steering_angles = shuffle(self.samples, self.steering_angles)


class CenterImageGenerator(Sequence):
    def __init__(self, samples, batch_size, corr=0.0):
        self.batch_size = batch_size
        self.corr = corr

        self.samples = []
        self.steering_angles = []

        for sample in samples:
            self.samples.append(sample[0])
            angle = float(sample[3])
            self.steering_angles.extend(angles)

        self.steering_angles = np.array(self.steering_angles)
        self._len = len(self.steering_angles)

        self.samples, self.steering_angles = shuffle(self.samples, self.steering_angles)

    def __len__(self):
        return (self._len + self.batch_size - 1) // self.batch_size

    def preprocess(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[70: 140, :, 2]
        # image = cv2.resize(image, (320, 160))
        return np.expand_dims(image, axis=2)

    def __getitem__(self, item):
        x_batch = np.empty([self.batch_size, 70, 320, 1])
        y_batch = self.steering_angles[item : item+self.batch_size]
        i = 0
        for img_path in self.samples[item: item+self.batch_size]:
            x_batch[i] = self.preprocess(img_path)

        return x_batch, np.array(y_batch)

    def on_epoch_end(self):
        self.samples, self.steering_angles = shuffle(self.samples, self.steering_angles)
