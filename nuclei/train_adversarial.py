# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Deep Learning implementation for image segmentation.
"""

import sys

from prepare import set_trainability, get_gan, get_discriminator, get_generator, get_data, get_masks
from keras.callbacks import ModelCheckpoint, EarlyStopping

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

def train(GAN, G, D, X, Y, epochs=(args.epochs or 20), n_samples=570, batch_size=(args.batch_size or 8)):
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

Y_train = get_masks(args.label, resolution=(args.resolution,args.resolution))
X_train = get_data(args.train, args.train, resolution=(args.resolution,args.resolution))

logging.info("Cargando red discriminadora")
discriminator = get_discriminator()
logging.info("Cargando red discriminadora")
generator = get_generator()
logging.info("Cargando red adversaria")
GAN = get_gan(generator,discriminator)
train(GAN, generator, discriminator, X_train, Y_train)
