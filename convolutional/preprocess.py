# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Preprocessing data functions and other utilities.
"""

import pandas as pd
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def get_data(data_info="train",folder=None):
    """
    Itera sobre cada carpeta que contiene mascaras e imagenes completas. Unicamente devuelve arrays de numpy
    con las imagenes completas.
    """
    logging.info("Obteniendo datos de %s" %data_info)
    logging.info("Ruta: %s" %folder)
    images = list()
    id_list = list(os.walk(folder))[0][1]
    [images.append(img_to_array(load_img("{folder}/{uid}/images/{uid}.png".format(folder=folder,uid=img_id)))) for img_id in tqdm(id_list)]
    res = np.asarray(images, dtype=object)
    logging.info("Data shape: %s" %res.shape)
    return res

get_data(data_info='STAGE 1 TEST',folder="data/stage1/stage1_test/")
