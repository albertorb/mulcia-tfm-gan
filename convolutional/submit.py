# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Generate submission for kaggle competition.
"""

from keras.models import load_model
from prepare import get_data, get_test_resolutions, get_test_original_resolution, iou, iou_metric

from tqdm import tqdm
from skimage.transform import resize
from skimage.morphology import label
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Model, load_model
from keras.layers import Dense, Input, merge, Dropout, Lambda, BatchNormalization, LeakyReLU, PReLU
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import tf
from keras.initializers import glorot_uniform

import logging
logging.basicConfig(level=logging.INFO)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", help="Ruta del modelo entrenado", type=str)
parser.add_argument("--test-dir", help="Ruta de imagenes de test", type=str)
parser.add_argument("--resolution", help="Redimensionar test, usar solo un entero. e.g: 128 para 128x128", type=int)

args = parser.parse_args()

logging.info("Loading model and weights...")
model = load_model(args.model_dir)
logging.info("Done.")

test = get_data(args.test_dir,args.test_dir)

logging.info("Predicting from test")
predictions = model.predict(test)
logging.info("Done.")

resized_predictions = get_test_original_resolution(test, get_test_resolutions(args.test_dir))
logging.info(resized_predictions[0].shape)
