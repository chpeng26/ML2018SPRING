
# coding: utf-8

# In[17]:

import numpy as np
import pandas as pd
import math
import sys


# In[9]:

# load train_x and train_y
train_x = []
train_y = []
with open(sys.argv[1], 'r') as file:
    tmp = []
    file = file.readlines()[1:] #delete the first line
    for line in file:
        line = line.strip("\n")
        tmp = line.split(',')
#         tmp = tmp[0:10] + tmp [11:] #delete fnlwgt feature
        train_x.append(tmp)
with open(sys.argv[2], 'r') as file:
    for line in file:
        train_y.append([int(line)])
train_x = np.array(train_x, int)
train_y = np.array(train_y)

# load test_x
test_x = []
with open(sys.argv[3], 'r') as file:
    tmp = []
    file = file.readlines()[1:] #delete the first line
    for line in file:
        line = line.strip("\n")
        tmp = line.split(',')
#         tmp = tmp[0:10] + tmp [11:] #delete fnlwgt feature
        test_x.append(tmp)
test_x = np.array(test_x, int)


# In[10]:

# normalization
train_test_x = np.concatenate((train_x,test_x))
mu = (sum(train_test_x) / train_test_x.shape[0])
sigma = np.std(train_test_x, axis=0)
mu = np.tile(mu, (train_test_x.shape[0], 1))
sigma = np.tile(sigma, (train_test_x.shape[0], 1))
train_test_x_normed = (train_test_x - mu) / sigma

# Split to train, test again
train_x = train_test_x_normed[0:train_x.shape[0]]
test_x = train_test_x_normed[train_x.shape[0]:]
# train_x = train_test_x[0:train_x.shape[0]]
# test_x = train_test_x[train_x.shape[0]:]

train_np = np.hstack((train_x,train_y))

df = pd.DataFrame(train_np)
dff = pd.DataFrame(test_x)

used_df = df
#cross validation
train_df = used_df.sample(frac=0.9, random_state=2148)
valid_df = used_df.loc[~df.index.isin(train_df.index)]
test_df = dff
test_x = test_df

fnum = train_x.shape[1]
train_y = train_df.iloc[:,fnum]
train_x = train_df.iloc[:,0:fnum]
valid_y = valid_df.iloc[:,fnum]
valid_x = valid_df.iloc[:,0:fnum]

#select training data
used_x = np.array(train_x)
used_y = np.array(train_y)

# sigmoid function
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-9, 1-(1e-9))
# def _shuffle(x, y):
#     randomize = np.arange(len(x))
#     np.random.shuffle(randomize)
#     return (x[randomize], y[randomize])


# In[11]:

#calculate mu1 and mu2
fnum = train_x.shape[1]
mu1 = np.zeros((fnum,))
mu2 = np.zeros((fnum,))
count1 = 0
count2 = 0
for i in range(len(used_x)):
    if used_y[i] == 1:
        mu1 += used_x[i]
        count1 += 1 
    else:
        mu2 += used_x[i]
        count2 += 1
mu1 /= count1
mu2 /= count2


# In[12]:

#calculate sigma1 and sigma2
sigma1 = np.zeros((fnum, fnum))
sigma2 = np.zeros((fnum, fnum))
train_data_size = used_x.shape[0]
for i in range(len(used_x)):
    if used_y[i] == 1:
        sigma1 += np.dot(np.transpose([(used_x[i] - mu1)]),[(used_x[i] - mu1)]) / count1
    else:
        sigma2 += np.dot(np.transpose([(used_x[i] - mu2)]),[(used_x[i] - mu2)]) / count2
sigma1 /= count1
sigma2 /= count2
shared_sigma = (float(count1)/train_data_size) * sigma1 + (float(count2)/train_data_size) * sigma2


# In[21]:

# predict the answer
def predict(test_x, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inverse)
    x = test_x.T
    b = (-0.5) * np.dot(np.dot(mu1, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2, sigma_inverse),mu2)     + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    return y


# In[22]:

prob = predict(test_x, mu1, mu2, shared_sigma, count1, count2)


# In[24]:

pred = (prob > 0.5).astype(int)

answerfile = sys.argv[4]
with open(answerfile,'w') as file:
    file.write('id,label\n')
    for i in range(len(pred)):
        row_output = str(i+1) + "," + str(pred[i]) + "\n"
        file.write(row_output)
print("All Done")


# In[ ]:



