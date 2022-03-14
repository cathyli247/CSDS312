IMG_SIZE = 380
CHANNELS = 3
BATCH_SIZE = 4
SEED = 12345

PRE_TRAINED_MODEL_PATH = './models/efficientnetv2-l_notop.h5'
PRE_TRAINED_MODEL_URL = 'http://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-l_notop.h5'
SAVED_MODEL_PATH = './models/our_model.h5'

FREEZED_EPOCH = 1
UNFREEZED_EPOCH = 1

DR_TRAIN_DF_PATH = "./diabetic_retinopathy/train.csv"
DR_TRAIN_IMG_PATH = "./diabetic_retinopathy/train_images/"
