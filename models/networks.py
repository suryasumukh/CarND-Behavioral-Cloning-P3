from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout

from keras.optimizers import Adam


def lenet_model(input_shape):
    _input = Input(shape=input_shape)

    conv1 = Conv2D(filters=8, kernel_size=5, strides=1,
                   activation='relu')(_input)
    pool1 = MaxPooling2D(pool_size=2, strides=2,
                         padding='same')(conv1)

    conv2 = Conv2D(filters=18, kernel_size=5, strides=1,
                   activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=2, strides=2,
                         padding='same')(conv2)

    flatten = Flatten()(pool2)

    fc1 = Dense(units=120, activation='linear')(flatten)
    fc2 = Dense(units=84, activation='linear')(fc1)

    output = Dense(units=1, activation='linear')(fc2)

    model = Model(inputs=[_input], outputs=[output])
    return model
