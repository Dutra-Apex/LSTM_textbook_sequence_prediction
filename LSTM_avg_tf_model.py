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

# Fits the x_train and y_train data into the model
history = model.fit(x_train, y_train, batch_size = 2, epochs= 500, validation_data=(x_test,y_test))

results = model.predict(x_test)

# Plots the predicted results vs the actual results
plt.figure(figsize=(10,6))
plt.title('Predicted tf-idf (green) & Actual tf-idf (red) - Good sequence')
plt.xlabel('Section')
plt.ylabel('Tf-idf value')
plt.scatter(range(len(results)), results, c='g')
plt.scatter(range(len(y_test)), y_test, c='r')
plt.show()

