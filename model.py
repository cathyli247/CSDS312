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

class Model():
    def __init__(self):
        if PRE_TRAINED_MODEL_NAME not in os.listdir(MODEL_FOLDER_PATH):
            r = requests.get(PRE_TRAINED_MODEL_URL, allow_redirects=True)
            with open(PRETRAINED_MODEL_PATH, 'wb') as f:
                f.write(r.content)

    def build(self, metrics=['accuracy']):
        input_0 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        input_0 = layers.RandomFlip(seed=SEED)(input_0)
        input_0 = layers.RandomContrast(factor=0.1, seed=SEED)(input_0)
        input_0 = layers.RandomRotation(factor=(-0.45, 0.45), seed=SEED)(input_0)

        self.effnet = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights=PRETRAINED_MODEL_PATH,
                                                                    include_top=False,
                                                                    input_tensor=input_0,
                                                                    input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
        self.effnet.trainable = False
        effnet_no_top_block = self.effnet.layers[-4].output

        top_output_1 = self.build_top(effnet_no_top_block, 1, 5)
        top_output_2 = self.build_top(effnet_no_top_block, 2, 2)

        self.model = tf.keras.Model(inputs=self.effnet.input, outputs=[
                            top_output_1, top_output_2])

        for i in [-3, -2, -1]:
            self.model.layers[2*i-8].set_weights(self.effnet.layers[i].get_weights())
            self.model.layers[2*i-7].set_weights(self.effnet.layers[i].get_weights())

        self.model.build((None, IMG_SIZE, IMG_SIZE, CHANNELS))

        losses = {
            'output_1': 'categorical_crossentropy',
            'output_2': 'binary_crossentropy',
        }

        self.model.compile(loss=losses,
                        optimizer=mixed_precision.LossScaleOptimizer(optimizers.Adam(
                            learning_rate=0.01)),
                        metrics=metrics)


    def build_top(self, effnet_output, suffix, output_num):
        conv = layers.Conv2D.from_config(self.effnet.layers[-3].get_config())
        conv._name = f'top_conv_{suffix}'
        top_component = conv(effnet_output)
        bn = layers.BatchNormalization.from_config(
            self.effnet.layers[-2].get_config())
        bn._name = f'top_bn_{suffix}'
        top_component = bn(top_component)
        act = layers.Activation.from_config(self.effnet.layers[-1].get_config())
        act._name = f'top_activation_{suffix}'
        top_component = act(top_component)

        top_component = layers.GlobalAveragePooling2D()(top_component)
        top_component = layers.BatchNormalization()(top_component)
        top_component = layers.Dropout(0.5)(top_component)

        top_component = layers.Dense(
            output_num, activation='softmax', name=f'output_{suffix}')(top_component)
        return top_component


    def fit(self, train_dataset, validation_dataset, epoch, batch_size, extras=[]):
        earlystop = callbacks.EarlyStopping(patience=10,
                                            monitor='val_loss',
                                            min_delta=0.0005,
                                            restore_best_weights=True)
        callback = [earlystop]
        callback.extend(extras)
        return self.model.fit(train_dataset,
                        batch_size=batch_size,
                        epochs=epoch,
                        validation_data=validation_dataset,
                        validation_batch_size=batch_size,
                        callbacks=callback)

    def unfreeze(self, metrics=['accuracy']):
        for layer in self.model.layers[-29:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        losses = {
            'output_1': 'categorical_crossentropy',
            'output_2': 'binary_crossentropy',
        }

        self.model.compile(loss=losses,
                    optimizer=mixed_precision.LossScaleOptimizer(optimizers.Adam(
                        learning_rate=0.0001)),
                    metrics=metrics)

    def evaluate(self, data, batch_size):
        return self.model.evaluate(data, batch_size=batch_size)

    def predict(self, data, batch_size):
        return self.model.predict(data, batch_size=batch_size)
