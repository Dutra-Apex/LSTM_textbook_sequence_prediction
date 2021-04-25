import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Creates array with all the tf-idf values of the good_seq
good_seq_all_tf = []
placeholder = []

for i in range(0, len(good_seq)):
  for j in good_seq[i]:
    placeholder.append(M_OS[j])
  good_seq_all_tf.append(placeholder)
  placeholder = []

#print(len(good_seq_all_tf[0]))

#Converts into np array
good_seq_all_tf = np.array(good_seq_all_tf)

#Creates a sorted array with the highest tf-idf values
# The sorting will happen based on the first section of wach good sequence
sorted_tf = np.zeros((112,3,1074))
index_list = []
placeholder = []

#Uses argsort to return the index of the highest tf-idf values
for i in range(0, len(good_seq_all_tf)):
  index_list.append(good_seq_all_tf[i][0].argsort()[::-1])
  for j in range(0, len(index_list[0])):
    sorted_tf[i][0][j] = good_seq_all_tf[i][0][index_list[0][j]]
    sorted_tf[i][1][j] = good_seq_all_tf[i][1][index_list[0][j]]
    sorted_tf[i][2][j] = good_seq_all_tf[i][2][index_list[0][j]]
  index_list = []

sorted_tf = np.array(sorted_tf)

#print(sorted_tf[3][:2])


""" 
# Plots the sorted tf-idf of all sections
for i in range(len(good_seq)):
  plt.figure(figsize=(10,6))
  plt.title("Tf-idf value of the sorted terms")
  plt.xlabel('Term')
  plt.ylabel('Tf-idf value')
  plt.plot(sorted_tf[i][0], alpha = 0.5)
  plt.plot(sorted_tf[i][1], alpha = 0.5)
  plt.plot(sorted_tf[i][2], alpha = 0.5)
  plt.legend(['Section %s' % (good_seq[i][0]),'Section %s' % (good_seq[i][1]),'Section %s' % (good_seq[i][2])],loc='upper right')
  #plt.savefig(path + '/good_seq_plots/plot_%s.png' % (i))
  plt.show()
"""


# The dataset is composted of the X highest tf-idf values in the sorted_tf
dataset = np.zeros((len(sorted_tf), len(sorted_tf[0]), 500))

for i in range(0, len(sorted_tf)):
  for j in range(0, len(sorted_tf[1])):
    for k in range(0, dataset.shape[2]):
      dataset[i][j][k] = sorted_tf[i][j][k]
      
training_data_len = math.ceil(len(dataset) * .8)

#Reshapes data into a 2D aray for normalization
dataset_2D = dataset.reshape((dataset.shape[0], dataset.shape[1]*dataset.shape[2]))

#Normalizes the data
scaler = MinMaxScaler (feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset_2D)

#Returns scaled_data to oiginal 3D shape
scaled_data = scaled_data.reshape(dataset.shape[0],dataset.shape[1],dataset.shape[2])

# Creates a training data with 80% of the original data
train_data = scaled_data[0:training_data_len, :]

#Split the data into x_train and y_train:

x_train = [] #Independent variable (used to predict)
y_train = [] #Dependent variables (values that the LSTM will try to predict)

for i in range(0,training_data_len):
    x_train.append([train_data[i][0], train_data[i][1]])
    y_train.append(train_data[i][2])
    
    
# Converts training set into numpy arrays and check their shape
x_train, y_train = np.array(x_train), np.array(y_train)

# Utilizes the remaining 20% of the data for testing the model
test_data = scaled_data[training_data_len:]
x_test = []
y_test = []

for i in range(0,len(test_data)):
    x_test.append([test_data[i][0], test_data[i][1]])
    y_test.append(test_data[i][2])

x_test, y_test = np.array(x_test), np.array(y_test)

# Generates a LSTM model
# The model will utilize the terms in the first 2 sections of each good sequence 
# to predict the tf-idf values of the terms in the 3rd section.
model = Sequential()
model.add(LSTM((500), batch_input_shape=(22, 2, 500),  return_sequences=True))
model.add(LSTM((500), return_sequences=True))
model.add(LSTM((500), return_sequences=False))
model.compile(loss = 'mean_absolute_error', optimizer='adam', metrics='accuracy')
model.summary()
