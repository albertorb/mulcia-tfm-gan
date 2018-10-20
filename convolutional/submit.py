# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Generate submission for kaggle competition.
"""

from keras.models import load_model
from prepare import get_data, get_test_resolutions, get_test_original_resolution, iou

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
