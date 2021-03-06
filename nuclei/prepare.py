# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Preprocessing data functions and other utilities.
"""

import imageio
import pandas as pd
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO, format='[tfm-nuclei] - %(message)s')

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from skimage.transform import resize
from skimage.morphology import label
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Input, concatenate, Dropout, Lambda, BatchNormalization, LeakyReLU, PReLU
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import tf
from keras.initializers import glorot_uniform

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
    return get_convolutional_model(resolution)

def get_gan(generator,discriminator,resolution=(128,128)):
    """
    Implementación de red adversaria
    """
    GAN = Sequential([generator, discriminator], name='GAN')
    GAN.compile(optimizer=Adam(lr=1e4), loss='binary_crossentropy')
    GAN.summary()
    return GAN

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def get_test_ids(test_dir):
    """
    Devuelve los UIDs del conjunto de test.
    """
    logging.info("Creando lista de UIDs para test")
    return list(os.walk(test_dir))[0][1]

def apply_encoding(data):
    """
    Aplica una codificación a las máscaras.
    La iteración que se realiza sobre el rango permite identificar de forma única a cada
    núcleo de la máscara, aplicándose a cada uno la codificación run length.
    """
    logging.info("Codificando datos en formato Run Length")
    rle_masks = list()
    for uid, region in tqdm(data):
      for i in range(1, region.max() + 1):
        rle_masks.append([uid,run_length_encode(region == i)])
    return rle_masks

def run_length_encode(mask):
    """
    Condifica el conjunto de datos al formato run-length propuesto en la
    competición de kaggle. El valor de entrada, mask, debe ser un array de numpy
    que previamente haya sido tratado por la funcion label de skimage.morphology.
    Dicha función realiza una clusterización de regiones, lo cual servirá para
    identificar individualmente a los núcleos dentro de la máscara.
    La forma en la que identifica los clusteres, es asignado a cada pixel un valor
    en función del grupo al que pertenece.
    """
    dots = np.where(mask.T.flatten()==1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def get_predicted_mask_separated(uids,images):
    """
    Dado un dataset de predicción, devuelven un array donde cada elemento
    se corresponde con una máscara unitaria por nucleo. Para facilitar su uso
    posteriormente, se transforma en array de tuplas, siendo el primer elemento
    siempre el uid de la imagen. El valor 0.5 es un umbral para establecer
    cuando se considera positivo o negativo el pixel con respecto a su inclusión
    en la máscara.
    """
    logging.info("Separando máscaras de forma atómica")
    masks = [(id,label(img > 0.5)) for id,img in zip(uids,images)]
    logging.info("Completado")
    return masks

def load_trained_model(model_dir=None):
    """
    Devuelve instancia de un modelo exportado con sus pesos ya entrenados.
    """
    logging.info("Cargando modelo y pesos...")
    model = load_model(model_dir, custom_objects={'iou': iou})
    logging.info("Completado")
    return model

def get_test_original_resolution(predictions, resolutions):
    """
    Convierte las imágenes generadas por la red como predicciones a su resolución
    real.
    """
    logging.info("Devolviendo predicciones a su resolución original")
    resized_predictions = [resize(image.astype(np.float32),resolutions[index][1]) for index,image in enumerate(tqdm(predictions))]
    return resized_predictions

def get_test_resolutions(test_dir=None):
    """
    Devuelve una lista de tuplas ID de imagen y Resolución de la misma.
    Es necesario para devolver a su resolución original a las máscaras generadas
    por la red neuronal.
    """
    logging.info("Obteniendo resoluciones del conjunto de test")
    logging.info("Ruta {path}".format(path=test_dir))
    id_list = list(os.walk(test_dir))[0][1]
    img_path = "{folder}/{uid}/images/{uid}.png"
    resolutions = [(id,img_to_array(load_img(img_path.format(folder=test_dir,uid=id))).shape[:2]) for id in tqdm(id_list)]
    return resolutions

def get_data(data_info="train",folder=None, resolution=(128,128)):
    """
    Itera sobre cada carpeta que contiene mascaras e imagenes completas. Unicamente devuelve arrays de numpy
    con las imagenes completas.
    """
    logging.info("Obteniendo datos de %s" %data_info)
    logging.info("Ruta: %s" %folder)
    images = list()
    id_list = list(os.walk(folder))[0][1]
    [images.append(img_to_array(load_img("{folder}/{uid}/images/{uid}.png".format(folder=folder,uid=img_id)))) for img_id in tqdm(id_list)]
    logging.info("Redimensionando imágenes a {resolution}".format(resolution=resolution))
    images = [resize(image.astype(np.uint8),resolution) for image in tqdm(images) ]
    res = np.asarray(images, dtype=object) # da problemas de memoria con 256x256
    return res

def get_masks(mask_path, resolution=(128,128)):
    """
    Itera de forma individual sobre cada máscara y las agrupa por imagen completa para generar una mascara
    completa de todos los nucleos celulares de una unica imagen completa.
    Las mascaras son convertidas a binario, siendo el valor del pixel 0 si no
    se corresponde con un nucleo, y 1 en caso de que si.
    """
    logging.info("Cargando mascaras en array de numpy")
    logging.info("Ruta %s" %mask_path)
    id_list = list(os.walk(mask_path))[0][1]
    labeled_images = list()
    for id_ in tqdm(id_list):
        img_dir = "{path}/{id_}/masks/".format(path=mask_path, id_=id_)
        total_mask = np.zeros(resolution)
        for mask in list(os.walk(img_dir))[0][2]:
            img_mask = imageio.imread(img_dir + mask, pilmode='L')#.astype(np.float32)
            img_mask_resized = resize(img_mask, resolution).astype(np.float32)
            total_mask = total_mask + img_mask_resized
        total_mask[total_mask > 0] = 1
        labeled_images.append(total_mask)
    res = np.asarray(labeled_images, dtype=object).reshape(-1,resolution[0],resolution[1],1)
    return res

def get_convolutional_model(resolution=(128,128)):
  """
  Basado en la arquitectura de la U-Net. Los kernels de convolucion han sido reducidos
  porque las imagenes de nuestro conjunto de entrenamiento tienen aproximadamente
  la mitad de la resolucion de los ejempos de U-Net.
  """
  inputs = Input((*resolution,3))

  conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs) # 64 : 4 = 16
  conv1 = LeakyReLU(alpha=0.3)(conv1)
  conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)  # ""
  conv1 = LeakyReLU(alpha=0.3)(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1) # 128 : 4 = 32
  conv2 = LeakyReLU(alpha=0.3)(conv2)
  conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2) # ""
  conv2 = LeakyReLU(alpha=0.3)(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)# 256 : 4 = 64
  conv3 = LeakyReLU(alpha=0.3)(conv3)
  conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
  conv3 = LeakyReLU(alpha=0.3)(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)# 512 : 4 = 128
  conv4 = LeakyReLU(alpha=0.3)(conv4)
  conv4 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
  conv4 = LeakyReLU(alpha=0.3)(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(512, 3,padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = LeakyReLU(alpha=0.3)(conv5)
  conv5 = Conv2D(512, 3,padding = 'same', kernel_initializer = 'he_normal')(conv5)
  conv5 = LeakyReLU(alpha=0.3)(conv5)
  drop5 = Dropout(0.5)(conv5)

  up6 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  up6 = LeakyReLU(alpha=0.3)(up6)
  merge6 = concatenate([drop4,up6],  axis = 3)
  conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = LeakyReLU(alpha=0.3)(conv6)
  conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
  conv6 = LeakyReLU(alpha=0.3)(conv6)

  up7 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
  up7 = LeakyReLU(alpha=0.3)(up7)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = LeakyReLU(alpha=0.3)(conv7)
  conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
  conv7 = LeakyReLU(alpha=0.3)(conv7)

  up8 = Conv2D(64, 2,padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  up8 = LeakyReLU(alpha=0.3)(up8)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = LeakyReLU(alpha=0.3)(conv8)
  conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
  conv8 = LeakyReLU(alpha=0.3)(conv8)

  up9 = Conv2D(32, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  up9 = LeakyReLU(alpha=0.3)(up9)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = LeakyReLU(alpha=0.3)(conv9)
  conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = LeakyReLU(alpha=0.3)(conv9)
  conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = LeakyReLU(alpha=0.3)(conv9)
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

  model = Model(input = inputs, output = conv10)

  model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [iou])

  return model

def iou_metric(y_true_in, y_pred_in, print_table=False):
    """
    Métrica intersección sobre la unión.
    """
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    iou = intersection / union

    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def iou(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value
