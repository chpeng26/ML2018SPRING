import pandas as pd
import numpy as np
import time
import h5py
from keras.models import Model, load_model, Input
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,PReLU, AveragePooling2D, GlobalAveragePooling2D, Average
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


# load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_y_all = train_df['label']
train_x = np.array(train_df['feature'])
test_y = test_df['id']
test_x = np.array(test_df['feature'])

# label one-hot encoding
train_y_oneHot_all = np_utils.to_categorical(train_y_all)  
test_y_oneHot = np_utils.to_categorical(test_y)

pixel_list = []
pixel_test_list = []
for image in train_x:
    image = image.split(" ")
    pixel_list.append(image)
for image in test_x:
    image = image.split(" ")
    pixel_test_list.append(image)

#just normalize by divide 255
pixel_matrix_all = []
nor_fac = np.array(np.full(2304,255)).reshape(48,48)
for train_data in pixel_list:
    train_data = np.array(train_data,dtype=float).reshape(48,48)
    train_nor_data = np.divide(train_data, nor_fac)
    pixel_matrix_all.append(train_nor_data)

#just normalize by divide 255
pixel_test_matrix = []
for test_data in pixel_test_list:
    test_data = np.array(test_data,dtype=float).reshape(48,48)
    train_nor_data = np.divide(test_data, nor_fac)
    pixel_test_matrix.append(train_nor_data)

#just convert to np.array
pixel_matrix_all = np.array(pixel_matrix_all).reshape(len(pixel_matrix_all),48,48,1)
pixel_test_matrix = np.array(pixel_test_matrix).reshape(len(pixel_test_matrix),48,48,1)

#Sampling rate
frac = int(pixel_matrix_all.shape[0]*0.9)

#split validation dataset
pixel_matrix = pixel_matrix_all[0:frac]
pixel_valid_matrix = pixel_matrix_all[frac:]
train_y_oneHot = train_y_oneHot_all[0:frac]
valid_y_oneHot = train_y_oneHot_all[frac:]
print(pixel_matrix.shape)
# set input shape and model_input
input_shape = pixel_matrix[0,:,:,:].shape
model_input = Input(shape=input_shape)


# set model parameter
def compile_and_train(model, num_epochs): 
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
    filepath = 'output/weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='output/logs/', histogram_freq=0, batch_size=32)
    history = model.fit(x=pixel_matrix, y=train_y_oneHot, batch_size=32, epochs=num_epochs, verbose=2, callbacks=[checkpoint, tensor_board], validation_split=0.1)
    return history
# evaluate model_error
def evaluate_error(model):
    pred = model.predict(pixel_valid_matrix, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, valid_y_oneHot)) / valid_y_oneHot.shape[0]  
  
    return error
# Model4:Original-CNN
def origin_cnn(model_input):
    x = Conv2D(32, kernel_size=(4,4), activation='relu', padding='same')(model_input)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=(4,4), activation='relu', padding='same')(x)                                                     
    x = Dropout(0.35)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=(4,4), activation='relu', padding='same')(x)
    x = Dropout(0.35)(x)
    x = AveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=(4,4), activation='relu', padding='same')(x)                                                     
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=(4,4), activation='relu', padding='same')(x)                                                     
    x = AveragePooling2D()(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(256, kernel_size=(4,4), activation='relu', padding='same')(x)                                                     
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, kernel_size=(4,4), activation='relu', padding='same')(x)                                                     
    x = AveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(7, (1,1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    model = Model(model_input, x, name='origin_cnn')

    return model
# Model1:ConvPool-CNN-C
def conv_pool_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding =    'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='conv_pool_cnn')
    
    return model
# Model2: ALL-CNN-C
def all_cnn(model_input):
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
        
    model = Model(model_input, x, name='all_cnn')
    
    return model

# Model3: Network in Network CNN
def nin_cnn(model_input):
    
    #mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(model_input)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(7, (1, 1))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='nin_cnn')
    
    return model
# Initiate the model and start Training
'''
conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)
m1 = compile_and_train(conv_pool_cnn_model, num_epochs=20)
m2 = compile_and_train(all_cnn_model, num_epochs=20)
m3 = compile_and_train(nin_cnn_model, num_epochs=20)

origin_cnn_model = origin_cnn(model_input)
m4 = compile_and_train(origin_cnn_model, num_epochs=20)
'''
# Reinitiate the model and load the best saved weights

conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)
origin_cnn_model = origin_cnn(model_input)
#load the best saved weights

conv_pool_cnn_model.load_weights('output/weights/conv_pool_cnn.19-0.26.hdf5')
all_cnn_model.load_weights('output/weights/all_cnn.19-0.14.hdf5')
nin_cnn_model.load_weights('output/weights/nin_cnn.19-1.37.hdf5')
origin_cnn_model.load_weights('output/weights/origin_cnn.19-0.84.hdf5')

models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model, origin_cnn_model]

#Start ensemble-stacking
def ensemble(models, model_input):
    
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model

ensemble_model = ensemble(models, model_input)

conv_pool_cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
all_cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
nin_cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
origin_cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])
ensemble_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 

scores = []
score1 = conv_pool_cnn_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
score2 = all_cnn_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
score3 = nin_cnn_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
score5 = origin_cnn_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
score4 = ensemble_model.evaluate(pixel_valid_matrix, valid_y_oneHot)
print(score1)
print(score2)
print(score3)
print(score5)
print(score4)

'''  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))
model.save('./output/model/'+ modelName +'str(time.strftime("%H-%M))' + '.h5')
answerfile = 'output/answer/' +'CNN-answer-' + str(time.strftime("%H-%M-%S")) + '.csv'
with open(answerfile,'w') as file:
    file.write('id,label\n')
    for i in range(len(prediction)):
        row_output = str(i) + "," + str(prediction[i]) + "\n"
        file.write(row_output)
print("Test Done")
'''

