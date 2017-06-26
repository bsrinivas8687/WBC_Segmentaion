from model import get_model
from data import get_train_data
from all_params import WEIGHTS
from metrics import np_dice_coef

X_train, Y_train = get_train_data()
X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
X_train /= 255.0
Y_train /= 255.0

model = get_model(train=False)
model.summary()
model.load_weights(WEIGHTS)
Y_preds = model.predict(X_train, batch_size=3, verbose=1)
print "\n{:.06f}".format(np_dice_coef(Y_train, Y_preds))
