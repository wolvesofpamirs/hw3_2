from common import generate_altered_images
import sys,os
from PIL import Image
import PIL
import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
print(K.image_data_format())
PATH = os.path.dirname(os.path.realpath(__file__))

folder = os.path.join(PATH,'DATA')
height = 32
width = 32
nb_channel = 3

filename = 'data_batch_1'
with open(os.path.join(folder,filename),'rb') as fr:
    save = pickle.load(fr,encoding='iso-8859-1')
X = save['data']
X.shape = (len(X), nb_channel, height, width)
print(X.shape)
#X = X.transpose(0,2,3,1)
print(X.shape)
print(X[0].shape)
#rgb, x, y
#dict = {'theta': 60}
#dict = {'flip_vertical': True}
dict = {'flip_horizontal': True}
img_gen = ImageDataGenerator(data_format = "channels_first")
#img_gen.apply_transform(args)
#im = Image.fromarray(np.uint8(X[0]))
#im.save('test.jpg')
y = img_gen.apply_transform(X[0], dict)
#print(y.shape)
#y = np.array(y)
y = y.transpose(1, 2, 0)
im = Image.fromarray(np.uint8(y))
##im2 = im.rotate(60)
#im2.save('test0.jpg')
im.save('test18.jpg')

#generate_altered_images(folder)