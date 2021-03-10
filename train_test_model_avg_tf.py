# Creates a training data with 80% of the original data
train_data = scaled_data[0:training_data_len, :]
bad_train_data = bad_scaled_data[0:training_data_len, :]

#Split the data into x_train and y_train:

x_train = [] #Independent variable (used to predict)
y_train = [] #Dependent variables (values that the LSTM will try to predict)
x2_train = []
y2_train = []

for i in range(0,training_data_len):
    x_train.append([[train_data[i][0]], [train_data[i][1]]])
    y_train.append(train_data[i][2])
    x2_train.append([[bad_train_data[i][0]], [bad_train_data[i][1]]])
    y2_train.append(bad_train_data[i][2])

# Assure that they have the same length for training
print(len(y_train), len(x_train))
print(len(y2_train), len(x2_train))

#Convert both train sets to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x2_train, y2_train = np.array(x2_train), np.array(y2_train)

# Utilizes the remaining 20% of the data for testing the model
test_data = scaled_data[training_data_len:]
bad_test_data = bad_scaled_data[training_data_len:]
x_test = []
y_test = []
x2_test = []
y2_test = []


for i in range(0,len(test_data)):
    x_test.append([[test_data[i][0]], [test_data[i][1]]])
    y_test.append(test_data[i][2])
    x2_test.append([[bad_test_data[i][0]], [bad_test_data[i][1]]])
    y2_test.append(bad_test_data[i][2])
    
# Converts both test sets into numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)
x2_test, y2_test = np.array(x2_test), np.array(y2_test)
