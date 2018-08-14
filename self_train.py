import pickle
import sys,os
import numpy as np
from common import load
from common import load_sup


PATH = os.path.dirname(os.path.realpath(__file__))

folder = os.path.join(PATH,'DATA')

#LX,LY, XU = load(folder)
LX, LY = load_sup(folder)
#XU = []
#print(LX[0])
#print(XU[0])
train_size = (int)(4*len(LX)/5)
Xval = LX[train_size+1000:]
Yval = LY[train_size+1000:]
train_size = (int)(train_size/2)
Xtrain = LX[:train_size]
Ytrain = LY[:train_size]
##XU = np.r_[XU, Xval]
print(LX.shape)
print(LY.shape)
print(Xtrain.shape)
print(Ytrain.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import maximum
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from new_model import newModel
from new_model2 import newModel2
from keras import optimizers
from keras.models import load_model
from heapq import nlargest
#import keras
'''
model = load_model('my_model2.h5')
loss_and_metrics = model.evaluate(Xval, Yval, batch_size=128)
print("eval:")
print(loss_and_metrics)
'''
'''
#print(model.predict(np.array([XU[0]])))
Ytrain_edit = Ytrain.copy()
Xtrain_edit = Xtrain.copy()
n_class = 10
n_unlabel = len(XU)
for i in range(4):
    Eval = (model.predict(XU, batch_size=128))
    Eval_max = Eval.max(axis=1)
    #print(Eval_max)
    #print(Eval_max.shape)
    
    n_new_sample = (int)(n_unlabel/10)
    Confient_sorted = nlargest(n_new_sample, range(len(Eval_max)), key=lambda k: Eval_max[k])
    #print(XU[Confient_sorted].shape)
    #print(Xtrain_edit.shape)
    Xtrain_edit = np.r_[Xtrain_edit, XU[Confient_sorted]]
    z = np.zeros((n_new_sample, n_class))
    z[range(n_new_sample), Eval[Confient_sorted].argmax(axis=1)] = 1
    Ytrain_edit = np.r_[Ytrain_edit, z]
    ind = np.ones((len(XU)), bool)
    ind[Confient_sorted] = False
    XU = XU[ind]
    model.fit(Xtrain_edit, Ytrain_edit, epochs=60, batch_size=100)
    loss_and_metrics = model.evaluate(Xval, Yval, batch_size=128)
    print("eval:")
    print(loss_and_metrics)
#print(Confient_sorted)
#print(Eval_max[Confient_sorted])
'''

datagen = ImageDataGenerator(
    #featurewise_center=True,
    horizontal_flip = True,
    vertical_flip = True,
    #zoom_range = 0.5,
    #featurewise_std_normalization=True,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    data_format = "channels_first"
    )
    
datagen = ImageDataGenerator(
    #featurewise_center=True,
    horizontal_flip = True,
    data_format = "channels_first"
    )
#datagen = ImageDataGenerator(data_format = "channels_first")
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(Xtrain)

# fits the model on batches with real-time data augmentation:

#model = newModel2()

#model.save('model2_1000_drop_kernal_init.h5')
model = load_model('model2_1000_drop_kernal_init.h5')
print(model.count_params())
epochs = 5000
batch_size = 128
print(len(Xtrain) / batch_size)
#for i in range(100):
    #print('Epoch:' + i)
earlyStopping=EarlyStopping(monitor='val_loss', patience=3, min_delta=10**-3, mode='auto')
checkPoint = ModelCheckpoint('model2_4000_drop_kernal_init.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5)
#model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, validation_data=(Xval, Yval), verbose = 2)
model.fit_generator(datagen.flow(Xtrain, Ytrain, batch_size=batch_size),
                        steps_per_epoch=len(Xtrain) / batch_size, epochs=epochs, validation_data=(Xval, Yval), verbose = 2, callbacks = [checkPoint])
model.save('model2_4000_drop_kernal_init.h5')
#
'''
for i in range(10):
    model.fit(Xtrain, Ytrain, epochs=10, batch_size=256)
    model.save('my_model3.h5')
'''
#model2_1000_no_drop: in acc hit 0.9 after 115 epochs 0.99 after 140 0.999 after 160 1 after 160 valid acc about 0.4
#model2_1000_no_drop_kernal_init: in acc hit 0.9 after 103 epochs 0.99 after 140 0.999 after 160 1 after 160 valid acc about 0.4
#model2_1000_drop_kernal_init: in train acc about 0.87 valid acc about 0.39
#model2_4000_drop_kernal_init: in train acc about 0.87 valid acc about 0.39
##model3_no_drop_4000_kernal_init: in acc hit 0.9 after 160 epochs valid acc about 0.48
#model2_no_drop_4000_kernal_init: in acc hit 0.98 after 160 epochs valid acc about 0.49
#model2_1000_drop2_kernal_init: in train acc about 0.99 valid acc about 0.43
#model2_4000_drop2_kernal_init: in train acc about 0.92 valid acc about 0.54
#model3_4000_no_drop_kernal_init: in train acc about 0.93 valid acc about 0.52
print(model.count_params())
model.save('model2_4000_drop.h5')
