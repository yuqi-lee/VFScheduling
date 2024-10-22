from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = pd.read_csv('VM500.csv', usecols=[0,1,2])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

for i in range(len(data_normalized)):
    print("data_normalized{} : {}".format(i, data_normalized[i]))

print(data_normalized)