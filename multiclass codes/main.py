import os
import random as rn

import numpy as np
import pandas as pd
import tensorflow as tf

from model import build_model, fit_data, unfreeze_last_block
from parameters import *
from preprocessing import img_aug


print(tf.config.list_physical_devices('GPU'))

tf.config.optimizer.set_jit('autoclustering')
print(tf.config.optimizer.get_jit())

tf.keras.mixed_precision.set_global_policy('mixed_float16')

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)

rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def get_dr_datasets():
    train_df = pd.read_csv(DR_TRAIN_DF_PATH)
    # Add extension to id_code
    train_df['id_code'] = train_df['id_code'] + ".png"
    train_df['diagnosis'] = train_df['diagnosis'].to_numpy().reshape(
        len(train_df), 1).tolist()
    print(f"# Training images: {train_df.shape[0]}")

    return train_df


train_df = get_dr_datasets()
train_datagen = img_aug()

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = build_model()

model.summary()

freezed_model_path = os.path.join(MODEL_FOLDER_PATH, FREEZED_SAVED_MODEL_NAME)
unfreezed_model_path = os.path.join(
    MODEL_FOLDER_PATH, UNFREEZED_SAVED_MODEL_NAME)

parallel_freezed_batch_size = FREEZED_BATCH_SIZE * \
    mirrored_strategy.num_replicas_in_sync

history = fit_data(train_df, train_datagen, model,
                   FREEZED_EPOCH, parallel_freezed_batch_size, freezed_model_path)
model.load_weights(freezed_model_path)

with mirrored_strategy.scope():
    model = unfreeze_last_block(model)

parallel_unfreezed_batch_size = UNFREEZED_BATCH_SIZE * \
    mirrored_strategy.num_replicas_in_sync

history = fit_data(train_df, train_datagen, model,
                   UNFREEZED_EPOCH, parallel_unfreezed_batch_size, unfreezed_model_path)
model.load_weights(unfreezed_model_path)

# print(f"The Training Categorical Accuracy Score is: \
#     {round(model.evaluate(train_generator, batch_size=UNFREEZED_BATCH_SIZE)[1], 5)}")
