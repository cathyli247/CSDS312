IMG_SIZE = 380
CHANNELS = 3
SEED = 12345

DATA_FOLDER_PATH = './data'
MODEL_FOLDER_PATH = './models'

PRE_TRAINED_MODEL_URL = 'http://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-l_notop.h5'

PRE_TRAINED_MODEL_NAME = 'efficientnetv2-l_notop.h5'
FREEZED_SAVED_MODEL_NAME = 'freezed_model.h5'
UNFREEZED_SAVED_MODEL_NAME = 'unfreezed_model.h5'

FREEZED_BATCH_SIZE = 64
UNFREEZED_BATCH_SIZE = 8
FREEZED_EPOCH = 1
UNFREEZED_EPOCH = 1

DR_TRAIN_DF_PATH = "./diabetic_retinopathy/train.csv"
DR_TRAIN_IMG_PATH = "./diabetic_retinopathy/train_images/"
