
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import sys
import time
import matplotlib.pyplot as plt


# In[ ]:

in_csv = "train.csv"
data = pd.read_csv(in_csv, encoding="big5")
for i in range(len(data.columns)):
    for j in range(len(data)):
        if data.iloc[j, i] == "NR":
            data.iloc[j, i] = 0


# In[ ]:

featureList = ['PM2.5','PM10','SO2','NO2','O3','CO','AMB_TEMP']
f_pow = [1,1,1,1,1,1,1]
used_data = data.loc[data["測項"].isin(featureList)].reset_index(drop=True)

#used for normalization
flist = [[] for i in range(sum(f_pow))] #每個feature的value
fmean = [[] for i in range(sum(f_pow))]
fstd = [[] for i in range(sum(f_pow))]

for i in range(0, len(used_data), len(featureList)):
    tmp = used_data.iloc[i:i+len(featureList), :]
    cur_f_ix = 0
    for j in range(len(featureList)):
        cur_record = pd.to_numeric(tmp[tmp['測項'] == featureList[j]].iloc[0,3:27])
        for k in range(1, f_pow[j]+1):
            flist[cur_f_ix].append((cur_record ** k).tolist())
            cur_f_ix += 1

for i in range(len(flist)):
    flist[i] = np.array(flist[i]).flatten()
    fmean[i] = flist[i].mean()
    fstd[i] = flist[i].std()


# In[ ]:

def get_date(date, row_ix):
    if row_ix >= len(used_data):
        return 1
    else:
        date = int(date.iloc[row_ix, 0].split("/")[-1])
        return date

def construct_record(cur_row, s_ix, e_ix, rows):
    tt = pd.Series()
    if s_ix < e_ix:
        tmp = used_data.iloc[cur_row:cur_row+rows,:]
        for i in range(rows):
            cur_record = pd.to_numeric(tmp[tmp["測項"] == featureList[i]].iloc[0, s_ix:e_ix])
            for j in range(1, f_pow[i]+1):
                tt = tt.append(cur_record ** j)

        tt = tt.append(pd.Series([int(tmp[tmp["測項"] == "PM2.5"].iloc[0,e_ix])]))
    else:
        tmp = used_data.iloc[cur_row:cur_row+rows+len(featureList),:]
        for i in range(rows):
            cur_record = pd.to_numeric(tmp[tmp["測項"] == featureList[i]].iloc[0,s_ix:27].append(tmp[tmp["測項"] == featureList[i]].iloc[1,3:e_ix]))
            for j in range(1, f_pow[i]+1):
                tt = tt.append(cur_record ** j)

        tt = tt.append(pd.Series([int(tmp[tmp["測項"] == "PM2.5"].iloc[1,e_ix])]))
    return tt

cur_row = 0
cur_date = get_date(used_data, cur_row)
start_ix = 3
end_ix = start_ix + n
df = pd.DataFrame()

while True:
    if start_ix > 26:
        start_ix = 3
        cur_row += len(featureList)
        cur_date += 1
    elif end_ix > 26 and get_date(used_data, cur_row+len(featureList)) != cur_date + 1:
        start_ix = 3
        end_ix = start_ix + n
        cur_row += len(featureList)
        cur_date = 1
        if cur_row >= len(used_data):
            break
    elif end_ix > 26:
        end_ix = 3

    df = df.append(construct_record(cur_row, start_ix, end_ix, len(featureList)).reset_index(drop=True), ignore_index=True)
    start_ix += 1
    end_ix += 1
    
fnum = len(df.columns) - 1 #the number of feature
df_y = df.iloc[:,fnum]


# In[ ]:

# normalization
nor_df = (df.iloc[:, 0:n] - fmean[0]) / fstd[0]
cur_ix = n
for i in range(1, len(flist)):
    nor_df = nor_df.join((df.iloc[:, cur_ix:cur_ix+n] - fmean[i]) / fstd[i])
    cur_ix += n
dff = df.copy()
nor_df = nor_df.join(df_y)


# In[ ]:

used_df = df

#cross validation
train_df = used_df.sample(frac=0.9, random_state=2148)
valid_df = used_df.loc[~df.index.isin(train_df.index)]

# train_df = nor_df.sample(frac=0.9, random_state=2148)
# valid_df = nor_df.loc[~df.index.isin(train_df.index)]

train_y = train_df.iloc[:,fnum]
train_df = train_df.iloc[:,0:fnum]
valid_y = valid_df.iloc[:,fnum]
valid_df = valid_df.iloc[:,0:fnum]

dff = df.iloc[:,0:fnum]
nor_df = nor_df.iloc[:,0:fnum]


# In[ ]:

#select training data
used_train = dff
used_y = df_y
# used_train = train_df
# used_y = train_y

print("train rows: %d" % (len(used_train)))
print("feature num: %d" % (fnum))
print(featureList)


# In[ ]:

iteration = 100000
lr = 0.5
b = 0
w = np.full(fnum, 1/fnum)
lam = 0.001
# print(w)
print("lamda : %f" % (lam))

lr_b = 0
lr_w = np.full(fnum, 0.0)

err_hist_y = np.array([])
iter_x = np.array([])


# In[ ]:

print("Start Training")
for i in range(iteration):  
    b_grad = 0.0
    w_grad = np.full(fnum, 0.0)
    
    b_grad = sum(-2.0 * (used_y - b - np.dot(used_train, w))) #b_grad = g
    w_grad = -2.0 * np.dot((used_y - b - np.dot(used_train, w)), used_train)+(2.0 * lam * w.sum())
    lr_b += b_grad ** 2 #lr_b -> ∑(g**2)
    b = b - lr / np.sqrt(lr_b) * b_grad
    lr_w += w_grad ** 2 #lr_w -> ∑(g**2)
    w = w - lr / np.sqrt(lr_w) * w_grad
    print(i)
    print(b)
    print('---')
    if i % 1000 == 0:
        if len(used_train) < 5500:
            estim = b + np.dot(valid_df, w)
            ans = valid_y
            root_mean_err = np.sqrt((sum((ans - estim) ** 2) / len(ans)))
            if i >= 200:
                iter_x = np.append(iter_x, i)
                err_hist_y = np.append(err_hist_y, root_mean_err)
#             print("Iteration: %d Done! Err: %f" % (i, root_mean_err))
        else:
            estim = b + np.dot(valid_df, w)
            ans = valid_y
            root_mean_err = np.sqrt((sum((ans - estim) ** 2) / len(ans)))
            if i >= 200:
                iter_x = np.append(iter_x, i)
                err_hist_y = np.append(err_hist_y, root_mean_err)
#             print("Iteration: %d Done! Err: %f" % (i, root_mean_err)) 


# In[ ]:

# save model
b = np.array(b)
np.save('model/model_b_simple.npy',b)
np.save('model/model_w_simple.npy',w)

