import os
import gc
import numpy as np
import random as rn
import pandas as pd
import tensorflow as tf
from keras import callbacks
from parameters import *
from preprocessing import img_aug
from model import build_model, unfreeze_last_block

print(tf.config.list_physical_devices('GPU'))

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

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    x_col='id_code',
                                                    y_col='diagnosis',
                                                    directory=DR_TRAIN_IMG_PATH,
                                                    target_size=(
                                                        IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    subset='training')

val_generator = train_datagen.flow_from_dataframe(train_df,
                                                  x_col='id_code',
                                                  y_col='diagnosis',
                                                  directory=DR_TRAIN_IMG_PATH,
                                                  target_size=(
                                                      IMG_SIZE, IMG_SIZE),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical',
                                                  subset='validation')

model = build_model()

checkpoint = callbacks.ModelCheckpoint(
    SAVED_MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min')
earlystop = callbacks.EarlyStopping(
    patience=5, monitor='val_loss', min_delta=0.0005)
tensorboard = callbacks.TensorBoard(log_dir="./logs/freezed",
                                    update_freq=100,
                                    profile_batch='100, 500')

model.summary()

history = model.fit(train_generator,
                    batch_size=BATCH_SIZE,
                    epochs=FREEZED_EPOCH,
                    validation_data=val_generator,
                    validation_batch_size=BATCH_SIZE,
                    callbacks=[earlystop, checkpoint, tensorboard])

model.load_weights(SAVED_MODEL_PATH)

tf.keras.backend.clear_session()
_ = gc.collect()

print(tf.config.experimental.get_memory_info('GPU:0')['current'])

# model = unfreeze_last_block(model)

# history = model.fit(train_generator,
#                     batch_size=BATCH_SIZE,
#                     epochs=UNFREEZED_EPOCH,
#                     validation_data=val_generator,
#                     validation_batch_size=BATCH_SIZE,
#                     callbacks=[earlystop, checkpoint])

# model.load_weights(SAVED_MODEL_PATH)

# print(f"The Training Categorical Accuracy Score is: \
#     {round(model.evaluate(train_generator, batch_size=BATCH_SIZE)[1], 5)}")
