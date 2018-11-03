# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Preprocessing data functions and other utilities.
"""

import keras
from convolutional.prepare import get_model as get_convolutional_model
import logging
logging.basicConfig(level=logging.INFO, format='[tfm-nuclei] - %(message)s')

import warnings
warnings.filterwarnings("ignore")

def get_discriminator(resolution=(128,128)):
    """
    Implementación de la red discriminadora.
    """
    inputs = Input((*resolution,1))

    kernel_initializer = glorot_uniform(seed=1)

    conv1 = Conv2D(128, 1, activation='relu', padding = 'same', kernel_initializer = kernel_initializer)(inputs)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(64, 1, strides=(2,2), activation='relu', padding = 'same', kernel_initializer = kernel_initializer)(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(32, 1, strides=(2,2), activation='relu', padding = 'same', kernel_initializer = kernel_initializer)(conv2)
    conv3 = BatchNormalization()(conv3)

    conv4 = Conv2D(32, 1, strides=(2,2), activation='relu', padding = 'same', kernel_initializer = kernel_initializer)(conv3)
    conv4 = BatchNormalization()(conv4)

    conv5 = Conv2D(16, 1, strides=(4,4), activation='relu', padding = 'same', kernel_initializer = kernel_initializer)(conv4)
    conv5 = BatchNormalization()(conv5)

    conv6 = Conv2D(16, 1, strides=(2,2), activation='relu', padding = 'same', kernel_initializer = kernel_initializer)(conv5)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv2D(16, 1, strides=(4,4), activation='relu', padding = 'same', kernel_initializer = kernel_initializer)(conv6)
    conv7 = BatchNormalization()(conv7)

    outputs = Conv2D(1, 1, activation = 'sigmoid')(conv7)

    model = Model(input = inputs, output = outputs, name='Discriminator')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy')

    return model

def get_generator(resolution=(128,128)):
    """
    Implementación de red generadora
    """
    return get_convolutional_model()

def get_gan(resolution=(128,128)):
    """
    Implementación de red adversaria
    """
    GAN = keras.models.Sequential([generator, discriminator], name='GAN')
    GAN.compile(optimizer=Adam(lr=1e4), loss='binary_crossentropy')
    GAN.summary()
    return GAN

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
