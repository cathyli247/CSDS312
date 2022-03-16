import os
import random as rn

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import Sequential, callbacks, layers, optimizers, mixed_precision

from parameters import *

rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def build_model():
    r = requests.get(PRE_TRAINED_MODEL_URL, allow_redirects=True)
    with open(PRE_TRAINED_MODEL_PATH, 'wb') as f:
        f.write(r.content)

    effnet = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights=PRE_TRAINED_MODEL_PATH,
                                                                   include_top=False,
                                                                   input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

    effnet.trainable = False

    model = Sequential([
        effnet,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax', dtype='float32')
    ])

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=5,
        decay_rate=0.5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=mixed_precision.LossScaleOptimizer(optimizers.Adam(
                      learning_rate=lr_schedule)),
                  metrics=['categorical_accuracy'])

    return model


earlystop = callbacks.EarlyStopping(patience=5,
                                    monitor='val_loss',
                                    min_delta=0.0005)
tensorboard = callbacks.TensorBoard(log_dir="./logs/freezed",
                                    update_freq=100,
                                    profile_batch='1, 49')


def fit_data(data, datagen, model, epoch, batch_size, model_path):
    train_generator = datagen.flow_from_dataframe(data,
                                                  x_col='id_code',
                                                  y_col='diagnosis',
                                                  directory=DR_TRAIN_IMG_PATH,
                                                  target_size=(
                                                        IMG_SIZE, IMG_SIZE),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  subset='training')

    val_generator = datagen.flow_from_dataframe(data,
                                                x_col='id_code',
                                                y_col='diagnosis',
                                                directory=DR_TRAIN_IMG_PATH,
                                                target_size=(
                                                    IMG_SIZE, IMG_SIZE),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                subset='validation')

    checkpoint = callbacks.ModelCheckpoint(filepath=model_path,
                                           monitor='val_loss',
                                           save_best_only=True,
                                           mode='min')

    return model.fit(train_generator,
                     batch_size=batch_size,
                     epochs=epoch,
                     validation_data=val_generator,
                     validation_batch_size=batch_size,
                     callbacks=[earlystop, checkpoint, tensorboard])


def unfreeze_last_block(model):
    for layer in model.layers[-22:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=5,
        decay_rate=0.5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=mixed_precision.LossScaleOptimizer(optimizers.Adam(
                      learning_rate=lr_schedule)),
                  metrics=['categorical_accuracy'])
