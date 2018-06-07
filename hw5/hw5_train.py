
# coding: utf-8

# In[1]:

import numpy as np
import os
from keras.initializers import random_normal
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pk
import sys


# In[ ]:

data = {}
in_train_label = sys.argv[1]
train_X, train_Y = [],[]
with open(in_train_label, 'r') as f:
    for line in f:
        train_Y.append(line.strip('\n').split(' +++$+++ ')[0])
        train_X.append(line.strip('\n').split(' +++$+++ ')[1])
data['train_data'] = [train_X, train_Y]
# train_Y = np.array(train_Y)
import itertools
# training data preprocessing
# remove the duplicate char in term
for i in range(len(train_X)):
    train_X[i] = train_X[i].split(' ')
    for j in range(len(train_X[i])):
        train_X[i][j] = ''.join(ch for ch, _ in itertools.groupby(train_X[i][j]))
train_X_re = []
for i in range(len(train_X)):
    tmp = ' '.join(train_X[i])
    train_X_re.append(tmp)


# In[ ]:

in_train_nolabel = sys.argv[2]
train_nolabel_X = []
with open(in_train_nolabel, 'r') as f:
    for line in f:
        train_nolabel_X.append(line.strip('\n'))

# remove the duplicate char in term
for i in range(len(train_nolabel_X)):
    train_nolabel_X[i] = train_nolabel_X[i].split(' ')
    for j in range(len(train_nolabel_X[i])):
        train_nolabel_X[i][j] = ''.join(ch for ch, _ in itertools.groupby(train_nolabel_X[i][j]))
train_nolabel_X_re = []
for i in range(len(train_nolabel_X)):
    tmp = ' '.join(train_nolabel_X[i])
    train_nolabel_X_re.append(tmp)
data['nolabel_data'] = [train_nolabel_X_re]


# In[ ]:

vocab_size = 20000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_X_re)

#save the tokenizer
with open ('output/pickle/tokenizer.pk', 'wb') as file:
    pk.dump(tokenizer, file)


# In[ ]:

# to_sequence
tmp = tokenizer.texts_to_sequences(train_X_re)
train_X_re = pad_sequences(tmp, maxlen=40)


# In[ ]:

model_inputs = Input(shape = (40,))


# In[ ]:

def simpleRNN4(model_inputs):
    embedding_inputs = Embedding(vocab_size, 128, trainable=True, embeddings_initializer=random_normal(seed=2, stddev=1.5))(model_inputs)
    RNN_outputs = Bidirectional(LSTM(256, return_sequences= True,dropout= 0.4,
                                     recurrent_dropout=0.2))(embedding_inputs)
    RNN_outputs = Bidirectional(LSTM(128, dropout = 0.4, recurrent_dropout=0.2))(RNN_outputs)
    outputs = Dense(256, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.1))(RNN_outputs)
    outputs = Dropout(0.3)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
    
    model = Model(inputs=model_inputs, outputs=outputs)
    return model


# In[ ]:

model0 = simpleRNN(model_inputs)
model0.load_weights('model/model_0.h5')
model5 = simpleRNN4(model_inputs)
model5.load_weights('model/model_5.h5')
model6 = simpleRNN5(model_inputs)
model6.load_weights('model/model_6.h5')
model7 = simpleRNN6(model_inputs)
model7.load_weights('model/model_7.h5')

tmp3 = tokenizer.texts_to_sequences(train_nolabel_X_re)
train_nolabel_X_re = pad_sequences(tmp3, maxlen=40)

model0_acc = 0.8033
model5_acc = 0.8132
model6_acc = 0.8107
model7_acc = 0.8142

total = 0.8033 + 0.8142 + 0.8107 + 0.8142

model0_rate = model0_acc/total
model5_rate = model5_acc/total
model6_rate = model6_acc/total
model7_rate = model7_acc/total

semi_pred0 = model0.predict(train_nolabel_X_re, batch_size=1024, verbose=True)

semi_pred5 = model5.predict(train_nolabel_X_re, batch_size=1024, verbose=True)

semi_pred6 = model6.predict(train_nolabel_X_re, batch_size=1024, verbose=True)

semi_pred7 = model7.predict(train_nolabel_X_re, batch_size=1024, verbose=True)

final_pred = (model0_rate * semi_pred0 + model5_rate * semi_pred5 + model6_rate* semi_pred6 + model7_rate * semi_pred7)


# In[ ]:

def get_semi_data(data,name,label,threshold,loss_function) : 
    # if th==0.3, will pick label>0.7 and label<0.3
    label = np.squeeze(label)
    index = (label>1-threshold) + (label<threshold)
    semi_X = train_nolabel_X_re
    semi_Y = np.greater(label, 0.5).astype(np.int32)
    if loss_function=='binary_crossentropy':
        return semi_X[index,:], semi_Y[index]
    elif loss_function=='categorical_crossentropy':
        return semi_X[index,:], to_categorical(semi_Y[index])
    else :
        raise Exception('Unknown loss function : %s'%loss_function)


# In[ ]:

model15 = simpleRNN4(model_inputs)

semi_X, semi_Y = get_semi_data(data,'nolabel_data', final_pred, 0.2, 'binary_crossentropy')
semi_X = np.concatenate((semi_X, train_X_re))
semi_Y = np.concatenate((semi_Y, train_Y))


# earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')
adam = Adam()
model15.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
filepath="model/model15_weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, 
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             monitor='val_acc',
                             mode='max' )
history = model15.fit(semi_X, semi_Y, 
                        validation_split=0.1,
                        epochs=30, 
                        batch_size=512,
                        callbacks=[checkpoint])

