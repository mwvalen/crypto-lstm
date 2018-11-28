import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_model(inputs, output_size, neurons,
                dropout =0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2]), activation="linear", return_sequences=True))
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2]), activation="linear", return_sequences=True))
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2]), activation="linear", return_sequences=True))
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2]), activation="linear"))
    model.add(Dropout(dropout))

    model.add(Dense(units=output_size))

    model.compile(loss=loss, optimizer=optimizer)
    return model

training_df = pd.read_csv("C:/Users/valen/Documents/MCP/BuyRate/Application/buy-rate/trainingData/ETH-USD-DD-14-56result.csv")

window_len = 15
result_forward_candles = 2
context_lookback = 5

training_df = training_df.drop(training_df.columns[2:4], axis=1)
training_df = training_df.drop(training_df.index[0:context_lookback-1])
training_df = training_df.drop(training_df.index[result_forward_candles*(-1):])
trainingScaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
trainingScaler.fit(training_df[training_df.columns[0:2]])
training_df[training_df.columns[0:2]] = trainingScaler.transform(training_df[training_df.columns[0:2]])
#print(training_df.head(30))
#training_df[training_df.columns] = scaler.inverse_transform(training_df[training_df.columns])
#print(training_df.head(30))

seed = 155
np.random.seed(seed)

LSTM_training_inputs = []

for i in range(len(training_df)-window_len):
    temp_set = training_df.iloc[i:(i+window_len), 0:2].copy()
    LSTM_training_inputs.append(temp_set)



LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)



LSTM_training_outputs = training_df[training_df.columns[2:3]]
LSTM_training_outputs = LSTM_training_outputs.drop(LSTM_training_outputs.index[0:window_len])

outputScaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
outputScaler.fit(LSTM_training_outputs)
LSTM_training_outputs = outputScaler.transform(LSTM_training_outputs)

X_train, X_test, y_train, y_test = train_test_split(LSTM_training_inputs, LSTM_training_outputs, test_size=0.25, shuffle=True)

training_model = build_model(X_train, output_size=1, neurons = 20)

training_history = training_model.fit(X_train, y_train,
                                        epochs=10, batch_size=1, verbose=2)

training_evaluation = training_model.evaluate(X_test, y_test, verbose=0)
print('training eval',training_evaluation)


#prediction_X = X_test
#prediction_y = training_model.predict(X_test)
#d = {'diff': X_test[X_test.columns[0:1]],
#    'priceMove': X_test[X_test.columns[1:2]],
#     'predictedMove': }

prediction_results = training_model.predict(X_test)
prediction_results = outputScaler.inverse_transform(prediction_results)


#print(X_test)
#for i in range(len(X_test)):
    #print ('i', i)
    #print(trainingScaler.inverse_transform(X_test[i]))
    #print(prediction_results[i])


#print('prediction results', prediction_results)
#print(X_test[:,0])
#print(X_test[:,1:2])
#X_test_df = pd.DataFrame({'Diff':X_test[:,0:1],'PriceMove':X_test[:,1:2]})
#print(trainingScaler.inverse_transform(X_test_df))
#print(outputScaler.transform(prediction_results))
