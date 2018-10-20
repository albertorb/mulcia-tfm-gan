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
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Model, load_model
from keras.layers import Dense, Input, merge, Dropout, Lambda, BatchNormalization, LeakyReLU, PReLU
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import tf
from keras.initializers import glorot_uniform


def get_test_original_resolution(predictions, resolutions):
    """
    Convierte las imágenes generadas por la red como predicciones a su resolución
    real.
    """
    logging.info("Devolviendo predicciones a su resolución original")
    resized_predictions = [resize(image.astype(np.uint8),resolutions[index][1]) for index,image in enumerate(tqdm(predictions))]
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
    resolutions = [(id,img_to_array(load_img(img_path.format(folder=test_dir,uid=id))).shape) for id in tqdm(id_list)]
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
    res = np.asarray(images, dtype=object)
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

def get_model(resolution=(128,128)):
  """
  Basado en la arquitectura de la U-Net. Los kenerls de convolucion han sido reducidos
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
  merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
  conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = LeakyReLU(alpha=0.3)(conv6)
  conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
  conv6 = LeakyReLU(alpha=0.3)(conv6)

  up7 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
  up7 = LeakyReLU(alpha=0.3)(up7)
  merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
  conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = LeakyReLU(alpha=0.3)(conv7)
  conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
  conv7 = LeakyReLU(alpha=0.3)(conv7)

  up8 = Conv2D(64, 2,padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  up8 = LeakyReLU(alpha=0.3)(up8)
  merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
  conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = LeakyReLU(alpha=0.3)(conv8)
  conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
  conv8 = LeakyReLU(alpha=0.3)(conv8)

  up9 = Conv2D(32, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  up9 = LeakyReLU(alpha=0.3)(up9)
  merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
  conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = LeakyReLU(alpha=0.3)(conv9)
  conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = LeakyReLU(alpha=0.3)(conv9)
  conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = LeakyReLU(alpha=0.3)(conv9)
  conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  #   morpho = Lambda(lambda x: grey_dilation(grey_erosion(x,size=8),6))(conv10)

  model = Model(input = inputs, output = conv10)

  model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [iou])
  #model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = [iou])

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

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
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
