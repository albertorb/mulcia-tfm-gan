# encoding: utf8
__author__ = "alberto.rincon.borreguero@gmail.com"
"""
Deep Learning implementation for image segmentation.
"""
from prepare import get_data, get_masks, get_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import logging
logging.basicConfig(level=logging.INFO)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", help="Set the batch size")
parser.add_argument("--epochs", help="Set number of epochs")
args = parser.parse_args()



Y_train = get_masks('data/stage1/stage1_train/')
X_train = get_data('TRAIN STAGE 1', 'data/stage1/stage1_train')

model = get_model()

epochs = args.epochs or 1
validation_size = 0.1
batch_size = args.batch_size or 8

early_stopping = EarlyStopping(monitor='val_iou', min_delta=0, patience=6, verbose=1, mode='max')
checkpoint = ModelCheckpoint('not_augmented_%sepochs_%fvalidationsize_%sbatchsize.hdf5' %(epochs,validation_size, batch_size),
                            monitor='val_iou', verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint, early_stopping]

_ = model.fit(x=X_train, y=Y_train, validation_split=validation_size, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1)
