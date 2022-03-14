import os
import gc
import numpy as np
import random as rn
import pandas as pd
import tensorflow as tf
from keras import callbacks
from parameters import *
from preprocessing import img_aug
from model import build_model, unfreeze_last_block, fit_data

print(tf.config.list_physical_devices('GPU'))

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

model = build_model()
model.summary()

history = fit_data(train_df, train_datagen, model,
                   FREEZED_EPOCH, FREEZED_BATCH_SIZE, FREEZED_SAVED_MODEL_PATH)
model.load_weights(FREEZED_SAVED_MODEL_PATH)

unfreeze_last_block(model)

history = fit_data(train_df, train_datagen, model,
                   UNFREEZED_EPOCH, UNFREEZED_BATCH_SIZE, UNFREEZED_SAVED_MODEL_PATH)
model.load_weights(UNFREEZED_SAVED_MODEL_PATH)

# print(f"The Training Categorical Accuracy Score is: \
#     {round(model.evaluate(train_generator, batch_size=UNFREEZED_BATCH_SIZE)[1], 5)}")
