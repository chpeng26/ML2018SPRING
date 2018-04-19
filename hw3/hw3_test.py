import pandas as pd
import numpy as np
import sys
from keras.utils import np_utils
from keras.models import load_model
test_df = pd.read_csv(sys.argv[1])

test_y = test_df['id']
test_x = np.array(test_df['feature'])

# label one-hot encoding  
test_y_oneHot = np_utils.to_categorical(test_y)

pixel_test_list = []

for image in test_x:
    image = image.split(" ")
    pixel_test_list.append(image)
nor_fac = np.array(np.full(2304,255)).reshape(48,48)
#just normalize by divide 255
pixel_test_matrix = []
for test_data in pixel_test_list:
    test_data = np.array(test_data,dtype=float).reshape(48,48)
    train_nor_data = np.divide(test_data, nor_fac)
    pixel_test_matrix.append(train_nor_data)

pixel_test_matrix = np.array(pixel_test_matrix).reshape(len(pixel_test_matrix),48,48,1)
# Making prediction and save result to prediction
model = load_model("model_2350.h5")
prediction = model.predict_classes(pixel_test_matrix)


#just convert to np.array
pixel_test_matrix = np.array(pixel_test_matrix).reshape(len(pixel_test_matrix),48,48,1)
answerfile = sys.argv[2]
with open(answerfile,'w') as file:
    file.write('id,label\n')
    for i in range(len(prediction)):
        row_output = str(i) + "," + str(prediction[i]) + "\n"
        file.write(row_output)
print("Test Done")