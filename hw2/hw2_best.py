
# coding: utf-8

# In[4]:

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import math
import time
import sys


# In[5]:

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


# In[32]:

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


# In[33]:

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


# In[34]:

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


# In[3]:

X = used_x
y = used_y
clf = MLPClassifier(activation='logistic',solver='sgd',shuffle= True ,alpha=1e-5,verbose=False,
                    hidden_layer_sizes=(125), random_state=1, tol=1e-5,batch_size=1,
                    learning_rate = "adaptive", early_stopping=True)


# In[43]:

clf.fit(X, y)


# In[46]:

pred = clf.predict(test_x)


# In[47]:

answerfile = sys.argv[4]
with open(answerfile,'w') as file:
    file.write('id,label\n')
    for i in range(len(pred)):
        row_output = str(i+1) + "," + str(int(pred[i])) + "\n"
        file.write(row_output)
print("All Done")

# In[ ]:



