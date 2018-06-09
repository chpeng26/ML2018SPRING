
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import pickle as pk
import sys
from keras import backend as K
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import random_normal


# In[ ]:

def mf(n_users, n_movies, latent_dim =128):
    user_input = Input(shape=[1])
    movie_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim,
                         embeddings_initializer=random_normal(stddev=0.01,seed=2))(user_input)
    user_vec = Flatten()(user_vec)
    movie_vec = Embedding(n_movies, latent_dim,
                          embeddings_initializer=random_normal(stddev=0.01,seed=2))(movie_input)
    movie_vec = Flatten()(movie_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    movie_bias = Embedding(n_movies, 1, embeddings_initializer='zeros')(movie_input)
    movie_bias = Flatten()(movie_bias)
    r_hat = Dot(axes=1)([user_vec, movie_vec])
    r_hat = Add()([r_hat, user_bias, movie_bias])
    model = Model([user_input, movie_input], r_hat)
    model.compile(loss='mse', optimizer='adam')
    return model


# In[18]:

def predict(testpath):
    test_df = pd.read_csv(testpath)
    test_all = test_df.values
    test_users, test_movies = [],[]
    for i in range(len(test_all)):
        test_users.append(test_all[i][1])
        test_movies.append(test_all[i][2])
    test_users = np.array(test_users)
    test_movies = np.array(test_movies)
    mf_model = mf(6040, 3952)
    mf_model.load_weights('mf_model11-0.5822.h5')
    predict = mf_model.predict([test_users, test_movies], verbose=0, batch_size=32)
    return predict


# In[19]:
testpath = sys.argv[1]
pred_nor = predict(testpath)


# In[5]:

with open ('ratings.pk','rb') as file:
    ratings = pk.load(file)
mean = np.mean(ratings)
std = np.std(ratings)


# In[21]:

pred = (pred_nor * std)+ mean
pred = np.clip(pred, 1, 5)


# In[22]:

answerfile = sys.argv[2]
with open(answerfile, 'w') as file:
    file.write('TestDataID,Rating\n')
    for i in range(len(pred)):
        output = str(i+1) + ',' + str(pred[i][0]) + '\n'
        file.write(output)

