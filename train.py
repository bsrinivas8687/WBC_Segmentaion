import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from model import get_model
from data import get_train_data
from metrics import dice_coef, dice_coef_loss
from all_params import *

X_train, Y_train = get_train_data()
X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
X_train /= 255.0
Y_train /= 255.0

model = get_model()
model.summary()
model.compile(optimizer=Adam(lr=BASE_LR), loss=dice_coef_loss, metrics=[dice_coef])
callbacks = [ModelCheckpoint(MODEL_CHECKPOINT_DIR + '{epoch:02d}_{loss:.06f}.hdf5', monitor='loss', save_best_only=True),
             ReduceLROnPlateau(monitor='loss', factor=0.1, patience=PATIENCE, min_lr=1e-07, verbose=1)]
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks)
model.save_weights(WEIGHTS)
