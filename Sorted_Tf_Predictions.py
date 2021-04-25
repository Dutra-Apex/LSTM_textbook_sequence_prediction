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


