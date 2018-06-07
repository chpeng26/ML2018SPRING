
# coding: utf-8

# In[1]:

import sys
import numpy as np
import itertools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Embedding, Dropout
from keras.initializers import random_normal
from keras import regularizers
import pickle as pk


# In[ ]:

#load the tokenizer
with open('./tokenizer.pk', 'rb') as file:
    tokenizer = pk.load(file)
model_inputs = Input(shape = (40,))
def simpleRNN4(model_inputs):
    embedding_inputs = Embedding(20000, 128, trainable=True, embeddings_initializer=random_normal(seed=2, stddev=1.5))(model_inputs)
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

in_test = sys.argv[1]
test_X, test_id = [], []
with open(in_test, 'r')as f:
    for line in f:
        if not line.startswith('i'):
            test_id.append(line.split(',')[0])
            test_X.append(line.strip('\n').split(',',1)[1])
# data['test_data'] = [test_X, test_id]

for i in range(len(test_X)):
    test_X[i] = test_X[i].split(' ')
    for j in range(len(test_X[i])):
        test_X[i][j] = ''.join(ch for ch, _ in itertools.groupby(test_X[i][j]))

test_X_re = []
for i in range(len(test_X)):
    tmp = ' '.join(test_X[i])
    test_X_re.append(tmp)


# In[ ]:

tmp2 = tokenizer.texts_to_sequences(test_X_re)
test_X_re = pad_sequences(tmp2, maxlen=40)

model15 = simpleRNN4(model_inputs)
model15.load_weights('./model15_weights-improvement-19-0.8336.hdf5')

prob = model15.predict(test_X_re, verbose=1)

pred = (prob > 0.5).astype(int)

ans_file = sys.argv[2]
with open(ans_file, 'w') as file:
    file.write('id,label\n')
    for i in range(len(pred)):
        output = str(i) + ',' + str(pred[i][0]) + '\n'
        file.write(output)

