# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Generate submission for kaggle competition.
"""

import pandas as pd
import numpy as np

from prepare import get_data, get_test_resolutions, get_test_original_resolution,\
                    iou, iou_metric, load_trained_model, get_predicted_mask_separated,\
                    apply_encoding, get_test_ids

import logging
logging.basicConfig(level=logging.INFO, format='[tfm-nuclei] - %(message)s')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", help="Ruta del modelo entrenado", type=str)
parser.add_argument("--test-dir", help="Ruta de imagenes de test", type=str)
parser.add_argument("--resolution", help="Redimensionar test, usar solo un entero. e.g: 128 para 128x128", type=int)

args = parser.parse_args()

model = load_trained_model(args.model_dir)

test = get_data(args.test_dir,args.test_dir)

logging.info("Realizando predicciones sobre el conjunto de test")
predictions = model.predict(test)
logging.info("Completado")

resized_predictions = get_test_original_resolution(predictions, get_test_resolutions(args.test_dir))
individualized_masks_with_uid = get_predicted_mask_separated(get_test_ids(args.test_dir) ,resized_predictions)
encoded_masks = apply_encoding(individualized_masks_with_uid)

submission = pd.DataFrame(np.array(encoded_masks, dtype=object), columns=["ImageId", "EncodedPixels"])
logging.info(submission.head())
