
# coding: utf-8

# In[8]:

import numpy as np
import os
from skimage import data, io, filters
from PIL import Image
import sys
# In[14]:
in_dir = sys.argv[1]
#in_dir = './Aberdeen/'
all_img_list= []
img_list = next(os.walk(in_dir))[2]
img_list = [os.path.join(in_dir, file) for file in img_list]
for image in img_list:
    img = io.imread(image)
    img = img.flatten()
    all_img_list.append(img)


# In[21]:
X = np.array(all_img_list)
# 1-1
X_mean = np.mean(X,axis=0)
#io.imsave('mean.jpg',np.reshape(X_mean,(600,600,3)).astype(np.uint8))

# In[19]:

# svd
U, s, V = np.linalg.svd((X - X_mean).T, full_matrices=False)

# 1-2

for i in range(4):
    eigen_img = 'eigenface_'+ str(i) + '.jpg' 
    M = U.T[i]
    M -= np.min(M)
    M /= np.max(M)
    M = M * -1
    M = (M*255).astype(np.uint8)
    #io.imsave(eigen_img, M.reshape(600,600,3).astype(np.uint8))

'''
eigen_img = 'eigenface_' + str(9) + '.jpg'
M = U.T[9]
M -= np.min(M)
M /= np.max(M)
M = M* -1
M = (M*255).astype(np.uint8)
io.imsave(eigen_img, M.reshape(600,600,3).astype(np.uint8))
'''
# 1-3
in_img = io.imread(in_dir + '/' + sys.argv[2])
#in_img = io.imread(sys.argv[2])
in_img = np.array(in_img)
in_img = in_img.flatten()
W = []
for i in range(4):
    W.append(np.dot((in_img-X_mean), U.T[i]))
#weights = np.dot((in_img-X_mean), U[:,:4])
#result = np.dot(weights, U[:,:4].T)
result = 0
for i in range(4):
    result += (W[i] * U.T[i])
result += X_mean
result -= np.min(result)
result /= np.max(result)
result = result * -1
result = (result*255).astype(np.uint8)
io.imsave('reconstruction.jpg', result.reshape(600,600,3).astype(np.uint8))
# 1-4 calculate the proportion
eigen_proportion = []
for i in range(4):
    s_proportion = s[i]/np.sum(s)
    eigen_proportion.append(s_proportion)
for j in range(len(eigen_proportion)):
    pass
    # print(eigen_proportion[j])
