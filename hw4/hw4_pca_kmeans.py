
# coding: utf-8

# In[18]:
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import numpy as np
import pandas as pd
import sys
#model_name = 'output/model/pca-kmean-'+ str(time.strftime("%d-%H-%M")) + '.h5'
# In[12]:

#build model
#input_img = Input(shape=cdcadadfasdf(784,))
pca = PCA(n_components=300, copy=True, whiten=True, random_state =6)
#load X
X = np.load(sys.argv[1])
X = X.astype('float32')/255.
frac = int(X.shape[0]*0.9)
X = np.reshape(X, (len(X), -1))
train_x = X[:frac]
val_x = X[frac:]

#do PCA
X_pca = pca.fit_transform(X)
#print(type(X_pca))
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_pca)

#read test cases
test_df = pd.read_csv(sys.argv[2])
IDs, idx1, idx2 = np.array(test_df['ID']), np.array(test_df['image1_index']), np.array(test_df['image2_index'])

# predict
answerfile = sys.argv[3]
with open(answerfile, 'w') as file:
    file.write("ID,Ans\n")
    for idx, i1, i2 in zip(IDs, idx1, idx2):
        p1 = kmeans.labels_[i1]
        p2 = kmeans.labels_[i2]
        if p1 == p2:
            pred = 1  # two images in same cluster
        else: 
            pred = 0  # two images not in same cluster
        file.write("{},{}\n".format(idx, pred))

