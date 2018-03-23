
# coding: utf-8

# In[1]:

import sys
import csv 
import math
import random
import numpy as np
import pandas as pd


# In[2]:

featureList = ['PM2.5','PM10','SO2','NO2','O3','CO','AMB_TEMP','NO']
f_pow = [1,1,1,1,1,1,1,1]
n = 9


# In[3]:

text = open(sys.argv[1] ,"r")
test = pd.read_csv(text, header=None)
for i in range(len(test.columns)):
    for j in range(len(test)):
        if test.iloc[j, i] == "NR":
            test.iloc[j, i] = 0
test_data = test.loc[test[1].isin(featureList)].reset_index(drop=True)


# In[4]:

# read model
b = np.load('model/model_b_strong.npy')
w = np.load('model/model_w_strong.npy')


# In[5]:

# save output file
filename = sys.argv[2]
with open(filename,'w') as file:
    file.write("id,value\n")
    counter = 0
    for i in range(0,len(test_data),len(featureList)):
        test_x = pd.Series()
        tmp = test_data.iloc[i:i+len(featureList),:]
        cur_f_ix = 0
        for j in range(len(featureList)):
            cur_record = pd.to_numeric(tmp[tmp[1] == featureList[j]].iloc[0,(11-n):11])
            for k in range(1, f_pow[j]+1):
                test_x = test_x.append(cur_record ** k)
                cur_f_ix += 1
        pred = b + np.dot(test_x, w)
        file.write("id_%d,%f\n" % (counter, pred))
        counter += 1


# In[ ]:



