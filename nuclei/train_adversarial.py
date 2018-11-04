# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Deep Learning implementation for image segmentation.
"""

import sys
import numpy as np

from functools import reduce
from copy import copy
from prepare import set_trainability, get_gan, get_discriminator, get_generator, get_data, get_masks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import logging
logging.basicConfig(level=logging.INFO, format='[tfm-nuclei] - %(message)s')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", help="Tamaño del batch", type=int)
parser.add_argument("--epochs", help="Número de épocas", type=int)
parser.add_argument("--resolution", help="Cambiar resolución de las imágenes durante entrenamiento. Sólo escribir una dimensión, ie: 128, para redimensionar a 128x128", type=int)
parser.add_argument("--train", help="Ruta a las imagenes de entrenamiento ", type=str)
parser.add_argument("--label", help="Ruta a las etiquetas de entrenamiento", type=str)
parser.add_argument("--export-dir", help="Ruta donde guardar los pesos generados ", type=str)

args = parser.parse_args()

def get_data_augmentation_generators(X_train, Y_train, batch_size=8):
    """
    """
    data_gen_args = dict(rotation_range=60,
                       width_shift_range=0.,
                       height_shift_range=0.,
                       zoom_range=0.)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    val_image_datagen = ImageDataGenerator(**data_gen_args)
    val_mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_datagen.fit(copy(np.array(X_train)[:600]), augment=True, seed=seed)
    mask_datagen.fit(copy(np.array(Y_train)[:600]), augment=True, seed=seed)

    val_image_datagen.fit(copy(np.array(X_train)[600:]), augment=True, seed=seed)
    val_mask_datagen.fit(copy(np.array(Y_train)[600:]), augment=True, seed=seed)

    image_generator = image_datagen.flow(copy(np.array(X_train)[:600]),seed=seed, batch_size=batch_size)

    mask_generator = mask_datagen.flow(copy(np.array(Y_train)[:600]),seed=seed, batch_size=batch_size)

    val_image_generator = val_image_datagen.flow(copy(np.array(X_train)[600:]),seed=seed, batch_size=batch_size)

    val_mask_generator = val_mask_datagen.flow(copy(np.array(Y_train)[600:]),seed=seed, batch_size=batch_size)

    train_generator = zip(image_generator, mask_generator)
    validation_generator = zip(val_image_generator, val_mask_generator)
    return train_generator, validation_generator

def get_batches_for_gan_augmented(train_generator, batch_size=32, n_samples=1000):
    """
    """
    x_examples = []
    y_examples = []
    end_loop = int(n_samples/batch_size)
    for i, (x,y) in enumerate(train_generator):
        if i == end_loop:
          break;
        x_examples.append(x)
        y_examples.append(y)
    return zip(np.array(x_examples), np.array(y_examples))



def train_gan_augmented(GAN, G, D, X_train, Y_train, epochs=(args.epochs or 20), n_samples=5000, batch_size=(args.batch_size or 8)):
    """
    """
    d_loss = []
    g_loss = []
    smooth_labels = 0
    epochs_range = range(epochs)
    train_generator, _ = get_data_augmentation_generators(X_train, Y_train, batch_size)
    #x_augmented, y_augmented = get_batches_for_gan_augmented(train_generator, batch_size, n_samples)
    for epoch in epochs_range:
        # Train discriminator
        # Do it separately for real and fake images
        #     batches = range(int(n_samples/batch_size))
        d_loss_batch = []
        for batch, (X,Y) in enumerate(get_batches_for_gan_augmented(train_generator, batch_size, n_samples)):
            half_batch = int(len(X) / 2)
            real_images, real_labels = Y[:half_batch], np.ones((half_batch, 1, 1, 1)) - smooth_labels # see: https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels
            fake_images, fake_labels = G.predict(X[half_batch:]), np.zeros((half_batch, 1, 1, 1)) + smooth_labels
            set_trainability(discriminator, True)
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            set_trainability(discriminator, False)
            d_loss.append(0.5 * np.add(d_loss_real, d_loss_fake))
            sys.stdout.write('\r'+"[Epoch %s] Batch %s -- (d_loss_real,d_loss_fake) - (%s,%s)" %(epoch, batch,d_loss_real,d_loss_fake))
            if (epoch % 3) == 0:
                g_loss.append(GAN.train_on_batch(np.array(X_train[batch*epoch:batch*epoch +batch_size]), np.zeros((batch_size, 1, 1, 1))))
        logging.info("\t\t perdida discriminador --> %s\n" %(d_loss[epoch]))
        if (epoch % 3) == 0:
          logging.info("\t\t perdida generador --> %s\n" %(g_loss[int(epoch/3)]))

        #d_loss.append(reduce(lambda x,y: (x+y)/2), d_loss_batch)




def train(GAN, G, D, X_train, Y_train, epochs=(args.epochs or 20), n_samples=570, batch_size=(args.batch_size or 8)):
    """
    Entrenamiento adversario
    """
    d_loss = []
    g_loss = []
    epochs_range = range(epochs)
    for epoch in epochs_range:
        batches = range(int(n_samples/batch_size))
        for batch in batches:
          half_batch = int(batch_size / 2)
          sys.stdout.write('\r'+"[Epoch %s] Batch %s de %s" %(epoch, batch, batches[-1]))
          if batch == batches[-1]:
            from_ = half_batch * epoch
            to_ = half_batch*epoch + half_batch
            real_images, real_labels = np.array(Y_train[-half_batch:]), np.array(np.zeros((half_batch, 1, 1, 1)))
            fake_images, fake_labels = generator.predict(np.array(X_train[-half_batch:])), np.array(np.ones((half_batch, 1, 1, 1)))
          else:
            from_ = half_batch * epoch
            to_ = half_batch*epoch + half_batch
            real_images, real_labels = np.array(Y_train[from_:to_]), np.zeros((half_batch, 1, 1, 1))
            fake_images, fake_labels = generator.predict(np.array(X_train[from_:to_])), np.array(np.ones((half_batch, 1, 1, 1)))

          set_trainability(discriminator, True)
          d_loss_real = discriminator.train_on_batch(real_images, real_labels)
          d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
          set_trainability(discriminator, False)
          d_loss.append(0.5 * np.add(d_loss_real, d_loss_fake))
          if (epoch % 3) == 0:
            g_loss.append(GAN.train_on_batch(np.array(X_train[batch*epoch:batch*epoch +batch_size]), np.zeros((batch_size, 1, 1, 1))))
        logging.info("\t\t perdida discriminador --> %s" %(d_loss[epoch]))
        if (epoch % 3) == 0:
          logging.info("\t\t perdida generador --> %s" %(g_loss[int(epoch/3)]))

Y_t = get_masks(args.label, resolution=(args.resolution,args.resolution))
X_t = get_data(args.train, args.train, resolution=(args.resolution,args.resolution))

logging.info("Cargando red discriminadora")
discriminator = get_discriminator()
logging.info("Cargando red discriminadora")
generator = get_generator()
logging.info("Cargando red adversaria")
GAN = get_gan(generator,discriminator)
#train(GAN, generator, discriminator, X_t, Y_t)
train_gan_augmented(GAN, generator, discriminator, X_t, Y_t)
