import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

# Load Data
Data_Frame = pd.read_csv('Electric_Production.csv', usecols= [1])
Data = Data_Frame.values

# Normalize Data
Scaler = MinMaxScaler(feature_range=(0, 1))
Data = Scaler.fit_transform(Data)

# Split Data
def Create_Data(Data, look_back= 1):
    X, Y = [], []
    for i in range(len(Data) - look_back):
        ans = Data[i : i + look_back, 0]
        X.append(ans)
        Y.append(Data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5
epochs = 20

Train_size = int(len(Data) * 0.6)
Test_size = len(Data) - Train_size
Train = Data[0 : Train_size, :]
Test = Data[Train_size : len(Data), :]

Train_Data, Train_Label = Create_Data(Train, look_back)
Test_Data, Test_Label = Create_Data(Test, look_back)

Train_Data = Train_Data.reshape(Train_Data.shape[0], 1, Train_Data.shape[1])
Test_Data = Test_Data.reshape(Test_Data.shape[0], 1, Test_Data.shape[1])

# Creat Model
Model = keras.Sequential()
Model.add(LSTM(128, input_shape= (1, look_back)))
Model.add(Dense(1))
Model.compile(loss= 'mean_squared_error', optimizer= 'adam')
Model.fit(Train_Data, Train_Label, epochs= epochs, batch_size= 1)

# Predict
Train_Predict = Model.predict(Train_Data)
Test_Predict = Model.predict(Test_Data)

Train_Predict = Scaler.inverse_transform(Train_Predict)
Train_Label = Scaler.inverse_transform([Train_Label])
Test_Predict = Scaler.inverse_transform(Test_Predict)
Test_Label = Scaler.inverse_transform([Test_Label])

# Visualize Result
Train_Predict_Plot = np.empty_like(Data)
Train_Predict_Plot[:, :] = np.nan
Train_Predict_Plot[look_back : look_back + len(Train_Predict), :] = Train_Predict

Test_Predict_Plot = np.empty_like(Data)
Test_Predict_Plot[:, :] = np.nan
Test_Predict_Plot[len(Train) + look_back : len(Train) + look_back + len(Test_Predict), :] = Test_Predict

fig = plt.figure('Visualize Result' ,figsize=(15, 9))

plt.plot(Scaler.inverse_transform(Data))
plt.plot(Train_Predict_Plot)
plt.plot(Test_Predict_Plot)

plt.show()

# Calculate Score
Score_Train = np.sqrt(mean_squared_error(Train_Label[0, :], Train_Predict[:, 0]))
print('Score Train is: ', Score_Train)
Score_Test = np.sqrt(mean_squared_error(Test_Label[0, :], Test_Predict[:, 0]))
print('Score Test is: ', Score_Test)