from keras.models import Sequential
from keras.layers import Dense
from keras.layers import maximum
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling1D
from keras.layers import Concatenate
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.optimizers import Adam


def newModel2():
    
    input_img = Input(shape=(3, 32, 32))
    #model = Sequential()
    
    x = Convolution2D(128, (7, 7),padding='same', data_format = "channels_first")(input_img)
    x = MaxPooling2D((2, 2 ), dim_ordering="th")(x)#, dim_ordering="th"
    s = Convolution2D(16, (1, 1),padding='same', data_format = "channels_first")(x)
    f1 = Convolution2D(64, (1, 1),padding='same', data_format = "channels_first")(s)
    f2 = Convolution2D(64, (3, 3),padding='same', data_format = "channels_first")(s)
    merge2 = Concatenate(axis=1)([f1, f2])
    #x = Convolution2D(64, (1, 1),padding='same', data_format = "channels_first")(x)
    #model.save('my_model2.h5')
    #x = BatchNormalization(epsilon=1e-03)(x)
    
    #x = (Dropout(0.5)(x))
    '''
    x = Convolution2D(64, (5, 5), dim_ordering="th",padding='same')(x)
    x = BatchNormalization(epsilon=1e-03)(x)
    x = MaxPooling2D((2, 2 ), dim_ordering="th")(x)
    '''
    x = Convolution2D(32, (5, 5),padding='same', data_format = "channels_first")(merge2)
    s = Convolution2D(16, (1, 1),padding='same', data_format = "channels_first")(x)
    f1 = Convolution2D(64, (1, 1),padding='same', data_format = "channels_first")(s)
    f2 = Convolution2D(64, (3, 3),padding='same', data_format = "channels_first")(s)
    merge2 = Concatenate(axis=1)([f1, f2])
    #x = Convolution2D(64, (1, 1),padding='same', data_format = "channels_first")(x)
    #x = BatchNormalization(epsilon=1e-03)(x)
    #x = MaxPooling2D((2, 2 ), dim_ordering="th")(x)
    #x = (Dropout(0.5)(x))
    '''
    x = Convolution2D(32, (3, 3),padding='same', data_format = "channels_first")(x)
    #x = BatchNormalization(epsilon=1e-03)(x)
    x = MaxPooling2D((2, 2 ), dim_ordering="th")(x)
    #x = (Dropout(0.5)(x))
    
    x = Convolution2D(32, (3, 3),padding='same', data_format = "channels_first")(x)
    #x = BatchNormalization(epsilon=1e-03)(x)
    x = MaxPooling2D((2, 2 ), dim_ordering="th")(x)
    '''
    #x = (Dropout(0.5)(x))
    #x = Convolution2D(128, 3, 3, dim_ordering="th")(x)
    #x = BatchNormalization(epsilon=1e-03)(x)
    #x = MaxPooling2D((2, 2 ), dim_ordering="th")(x)
    #model.add(Convolution2D(50, 3, 3, dim_ordering="th"))
    #model.add(MaxPooling2D((2, 2 ), dim_ordering="th"))
    
    x = (Flatten())(merge2)
    #print(x)
    #model.add(Dense(units=128, activation='linear'))
    #Dense(2, activation='linear')(x) 
    
    #maxout = maximum([(Dense(units=32, activation='linear')(x)) for _ in range(2)])
    maxout = (Dense(units=32, activation='relu')(x))
    #model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    x = BatchNormalization(epsilon=1e-03)(maxout)
    #x = (Dropout(0.6)(x))
    #model.add(Dense(units=128, activation='linear'))
    #model.add(MaxPooling1D(pool_size=2))
    #Dense(2, activation='linear')(x) 
    #maxout = (maximum([Dense(16, activation='linear')(x) for _ in range(2)]))
    '''
    maxout = (Dense(units=32, activation='relu')(x))
    x = BatchNormalization(epsilon=1e-03)(maxout)
    x = (Dropout(0.6)(x))
    '''
    #maxout = (maximum([Dense(16, activation='linear')(x) for _ in range(2)]))
    '''
    maxout = (Dense(units=32, activation='relu')(x))
    x = BatchNormalization(epsilon=1e-03)(maxout)
    x = (Dropout(0.6)(x))
    '''
    #model.add(Dense(units=128, activation='linear'))
    #model.add(MaxPooling1D(pool_size=2))
    #maxout = (maximum([Dense(16, activation='linear')(x) for _ in range(2)]))
    maxout = (Dense(units=32, activation='relu')(x))
    x = BatchNormalization(epsilon=1e-03)(maxout)
    #x = (Dropout(0.6)(x))
    
    
    '''
    maxout = (maximum([Dense(64, activation='linear')(x) for _ in range(2)]))
    x = (Dropout(0.3)(maxout))
    maxout = (maximum([Dense(64, activation='linear')(x) for _ in range(2)]))
    x = (Dropout(0.3)(maxout))
    '''
    #model.add(Dense(units=128, activation='relu'))
    #model.add(Dense(units=128, activation='relu'))
    #model.add(Dense(units=128, activation='relu'))
    #model.add(Dense(units=128, activation='relu'))
    #Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
    output = (Dense(units=10, activation='softmax')(x))
    model = Model(input_img, output)
    model.compile(loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])
    return model