
import pandas as pd
import numpy as np
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, PReLU, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

train_df = pd.read_csv(sys.argv[1])


train_y_all = train_df['label']
train_x = np.array(train_df['feature'])


# label one-hot encoding
train_y_oneHot_all = np_utils.to_categorical(train_y_all)  


pixel_list = []
pixel_test_list = []
for image in train_x:
    image = image.split(" ")
    pixel_list.append(image)

#just normalize by divide 255
pixel_matrix_all = []
nor_fac = np.array(np.full(2304,255)).reshape(48,48)
for train_data in pixel_list:
    train_data = np.array(train_data,dtype=float).reshape(48,48)
    train_nor_data = np.divide(train_data, nor_fac)
    pixel_matrix_all.append(train_nor_data)

#just convert to np.array
pixel_matrix_all = np.array(pixel_matrix_all).reshape(len(pixel_matrix_all),48,48,1)

#Sampling rate
frac = int(pixel_matrix_all.shape[0]*0.9)

#split validation dataset
pixel_matrix = pixel_matrix_all[0:frac]
pixel_valid_matrix = pixel_matrix_all[frac:]
train_y_oneHot = train_y_oneHot_all[0:frac]
valid_y_oneHot = train_y_oneHot_all[frac:]

# Training Model
model = Sequential()

# CNN layer
# layer
model.add(Conv2D(filters=32,kernel_size=3,input_shape=(48,48,1), activation='relu',padding='same'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D())
model.add(BatchNormalization())
#layer
model.add(Conv2D(filters=32,kernel_size=5, activation='relu',padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D())
#model.add(AveragePooling2D())
model.add(BatchNormalization())
#layer
model.add(Conv2D(filters=64,kernel_size=3, activation='relu',padding='same'))
model.add(Dropout(0.35))
model.add(BatchNormalization())
#layer
model.add(Conv2D(filters=64,kernel_size=5, activation='relu',padding='same'))                                              
model.add(Dropout(0.35))
model.add(AveragePooling2D())
model.add(BatchNormalization())
#layer
model.add(Conv2D(filters=128,kernel_size=3, activation='relu',padding='same'))                                                                  
model.add(Dropout(0.4))                                                                                                                                                                                                                                            
model.add(BatchNormalization())
#layer
model.add(Conv2D(filters=128,kernel_size=3, activation='relu',padding='same'))                                                                  
model.add(AveragePooling2D())
model.add(Dropout(0.4))
#layer
model.add(Conv2D(filters=256,kernel_size=3, activation='relu',padding='same'))                                                       
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(filters=256,kernel_size=5, activation='relu',padding='same'))                                                       
model.add(AveragePooling2D())
model.add(Dropout(0.5))                                                  
model.add(BatchNormalization())

model.add(Flatten())

#NN hidden layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(0.5))

#output layer
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

#checkpoint and earlystop
#filepath="output/weights_improvement-{epoch:02d}-{val_acc:.2f}"+str(time.strftime("%H-%M-%S"))+".hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc',save_best_only=True)
earlystop = EarlyStopping(monitor='acc',patience=3,mode='max')
# callbacks_list = [checkpoint,earlystop]
callbacks_list = [earlystop]
model.fit(pixel_matrix, train_y_oneHot, verbose=2, batch_size=32, shuffle=True, epochs = 100,validation_split=0.1,callbacks=callbacks_list)


