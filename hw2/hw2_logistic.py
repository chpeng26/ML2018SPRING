
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import math
import time
import sys


# In[2]:

# load train_x and train_y
train_x = []
train_y = []
with open(sys.argv[1], 'r') as file:
    tmp = []
    file = file.readlines()[1:] #delete the first line
    for line in file:
        line = line.strip("\n")
        tmp = line.split(',')
        tmp = tmp[0:10] + tmp [11:] #delete fnlwgt feature
        train_x.append(tmp)
with open(sys.argv[2], 'r') as file:
    for line in file:
        train_y.append([int(line)])
train_x = np.array(train_x, int)
train_y = np.array(train_y)


# In[3]:

# load test_x
test_x = []
with open(sys.argv[3], 'r') as file:
    tmp = []
    file = file.readlines()[1:] #delete the first line
    for line in file:
        line = line.strip("\n")
        tmp = line.split(',')
        tmp = tmp[0:10] + tmp [11:] #delete fnlwgt feature
        test_x.append(tmp)
test_x = np.array(test_x, int)


# In[4]:

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
train_np = np.hstack((train_x,train_y))


# In[5]:

df = pd.DataFrame(train_np)
dff = pd.DataFrame(test_x)

used_df = df
#cross validation
train_df = used_df.sample(frac=0.9, random_state=2148)
valid_df = used_df.loc[~df.index.isin(train_df.index)]
test_df = dff
test_x = test_df


# In[6]:

fnum = train_x.shape[1]
train_y = train_df.iloc[:,fnum]
train_x = train_df.iloc[:,0:fnum]
valid_y = valid_df.iloc[:,fnum]
valid_x = valid_df.iloc[:,0:fnum]


# In[7]:

#select training data
used_x = np.array(train_x)
used_y = np.array(train_y)


# In[8]:

# sigmoid function
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-9, 1-(1e-9))
# def sigmoid(z):
#     res = 1 / (1.0 + np.exp(-z))
#     return np.clip(res, 1e-9, 1-(1e-9))
def _shuffle(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    return (x[randomize-1], y[randomize-1])


# In[9]:

# Update parameter
# iteration = 10000
lr = 0.001
b = 0
w = np.full(fnum, 1/fnum)
# w = np.zeros((fnum))
lam = 0.01

err_hist_y = np.array([])
iter_x = np.array([])
train_data_size = len(train_x)
batch_size = 1
#best initial
best_b = b
best_w = w
best_acc = 0
pre_acc = 0

step_num = int(math.floor(train_data_size / batch_size))
epoch_num = 1000
save_param_iter = 10


# In[10]:

print("Start Training")
total_loss = 0.0
for epoch in range(0, epoch_num):
    if (epoch) % save_param_iter == 0:
        print('=====Saving Param at epoch %d=====' % epoch)
        print('epoch avg loss = %f' % (total_loss))
        if epoch !=0:
            iter_x = np.append(iter_x, epoch)
            err_hist_y = np.append(err_hist_y, total_loss)
        
        
        valid_data_size = len(valid_x)
        z = np.dot(valid_x, w) + b
        y = sigmoid(z)
        y_ = np.around(y)
        result = (np.squeeze(valid_y) == y_)
        acc = result.sum()/valid_data_size
        print('Validation acc = %f' % acc)
        
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_b = b
            best_count = 0
            best_loss = total_loss
            best_iter = epoch
        elif acc > pre_acc:
            best_count = 0
        else:
            if acc == best_acc:
                best_w = w
                best_b = b
                best_iter = epoch
            best_count += 1
            if best_count == 30:
                break
        pre_acc = acc
        total_loss = 0.0
        used_x , used_y = _shuffle(used_x, used_y)
        for idx in range (step_num):
            X = used_x[idx*batch_size:(idx+1)*batch_size]
            Y = used_y[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, w) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.mean(-1 * (np.squeeze(Y) - y))

            w = w - lr * w_grad
            b = b - lr * b_grad

# output file
prob = sigmoid(np.dot(test_x, best_w) + best_b)
pred = (prob > 0.5).astype(int)

answerfile = sys.argv[4]
with open(answerfile,'w') as file:
    file.write('id,label\n')
    for i in range(len(pred)):
        row_output = str(i+1) + "," + str(pred[i]) + "\n"
        file.write(row_output)
print("All Done")
