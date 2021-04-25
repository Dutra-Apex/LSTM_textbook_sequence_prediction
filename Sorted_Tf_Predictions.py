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

print(len(good_seq_all_tf[0]))
