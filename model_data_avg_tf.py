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

#Creates array with the average tf-idf values of the good sequences
good_seq_tf = []
bad_seq_tf = []
placeholder = []

for i in range(0, len(good_seq)):
  for j in good_seq[i]:
    placeholder.append(np.average(M_OS[j]))
  good_seq_tf.append(placeholder)
  placeholder = []

for i in range(0, len(bad_seq)):
  for j in bad_seq[i]:
    placeholder.append(np.average(M_OS[j]))
  bad_seq_tf.append(placeholder)
  placeholder = []


#Converts into np array
dataset = np.array(good_seq_tf)
bad_seq_dataset = np.array(bad_seq_tf)

#print(dataset[1:3])

# Uses 80% of the good_seq as the training data
training_data_len = math.ceil(len(dataset) * .8)
#print(training_data_len)

#Normalizes the data
scaler = MinMaxScaler (feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
bad_scaled_data = scaler.fit_transform(bad_seq_dataset)


