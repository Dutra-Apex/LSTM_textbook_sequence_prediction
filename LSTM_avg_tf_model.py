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

# Generates a LSTM model
model = Sequential()
model.add(LSTM((1), batch_input_shape=(None,2,1), return_sequences=True))
#model.add(LSTM((1), return_sequences=True))
model.add(LSTM((1), return_sequences=False))
model.compile(loss = 'mean_absolute_error', optimizer='adam')
model.summary()
