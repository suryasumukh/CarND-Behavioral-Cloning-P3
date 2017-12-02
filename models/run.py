from models import ImageGenerator
from models import read_samples

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint

import os


DATA_DIR = os.path.join(os.getcwd(), '../data')

samples = read_samples(DATA_DIR)
train_set, test_set = train_test_split(samples, test_size=0.2)
train_set, valid_set = train_test_split(train_set, test_size=0.2)

train_iterator = ImageGenerator(train_set, batch_size=256)
valid_iterator = ImageGenerator(valid_set, batch_size=256)


if __name__ == '__main__':
    # print(DATA_DIR)
    samples = read_samples(DATA_DIR)
    train_set, test_set = train_test_split(samples, test_size=0.2)
    train_set, valid_set = train_test_split(train_set, test_size=0.2)
