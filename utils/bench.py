import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


file = "500.csv"
directory_1 = '2013-8'
directory_2 = '2013-9'
directory_3 = '2013-7'
input_seq_length = 360
output_seq_length = 10
all_data = []
test_data = []

def read_train_data(directories):
    for directory in directories:
        filename = os.path.join(directory, file)
        df = pd.read_csv(filename, skiprows=1, sep=';\s*', engine='python')
        data = df.iloc[:, [4, 6, 10]].values  
        all_data.extend(data)

def read_test_data(directories):
    for directory in directories:
        filename = os.path.join(directory, file)
        df = pd.read_csv(filename, skiprows=1, sep=';\s*', engine='python')
        data = df.iloc[:, [4, 6, 10]].values  
        test_data.extend(data)

def plot_fig():
    plt.plot(all_data)
    plt.xlabel('Time')
    plt.ylabel('Network Bandwidth')
    #plt.title('Combined Data from Column 10')
    plt.show()
    plt.savefig('1.png')

def createXY(dataset, n_past, n_future):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset) - n_future):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i:i + n_future, dataset.shape[1]-1:dataset.shape[1]])
    return np.array(dataX), np.array(dataY)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_seq_length):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_seq_length = output_seq_length

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, 0:output_seq_length, ])
        return out


read_train_data([directory_1, directory_2])
read_test_data([directory_3])

scaler = MinMaxScaler(feature_range = (0,1))
df_for_training_scaled = scaler.fit_transform(all_data)
df_for_testing_scaled = scaler.fit_transform(test_data)

trainX, trainY = createXY(df_for_training_scaled, input_seq_length, output_seq_length)
testX, testY = createXY(df_for_testing_scaled, input_seq_length, output_seq_length)

input_dim = 3
output_dim = 1
hidden_size = 64
num_layers = 4

trainX = torch.tensor(trainX, dtype=torch.float32)
trainY = torch.tensor(trainY, dtype=torch.float32)
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)

model = LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, output_size=output_dim, output_seq_length=output_seq_length)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 12
for epoch in range(num_epochs):
    outputs = model(trainX)
    #print("outputs shape:", outputs.shape)
    loss = criterion(outputs, trainY)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

testX = torch.tensor(testX, dtype=torch.float32)
testY = torch.tensor(testY, dtype=torch.float32)
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)

# Make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(testX)

# Calculate the loss on the test data
test_loss = criterion(predictions, testY)
print(f'Test Loss: {test_loss.item():.4f}')

#predictions = torch.squeeze(predictions)
print("predictions shape:", predictions.shape)
print("predictions shape:", predictions[0])

# Plot the true values and predictions
real = df_for_testing_scaled[input_seq_length:len(test_data)-output_seq_length, 2]
pre = []
for i in range(0, len(predictions), output_seq_length):
    pre.extend(predictions[i, 0:output_seq_length, 0])


pre = [tensor.item() for tensor in pre]
#print("pre", pre)
print("real", real)
seq = list(range(len(real)))

plt.plot(real, label='Real')
plt.plot(seq, pre, label='Predicted')
plt.legend()
plt.show()
plt.savefig('1.png')

