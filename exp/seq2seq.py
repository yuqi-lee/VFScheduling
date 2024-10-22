import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

# Load data
data = pd.read_csv('VM500.csv')
data = data.values

# Define input sequence length, output sequence length and split ratio
input_seq_len = 300
output_seq_len = 300
split_ratio = 0.7
learning_rate = 0.001
early_stop_loss = 0.001
hidden_size = 100
epoch_num = 20

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split data into training set and test set
train_size = int(len(data) * split_ratio)
train, test = data[0:train_size], data[train_size:len(data)]

# Convert data into input/output pairs
def create_inout_sequences(input_data, tw_in, tw_out):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw_in-tw_out):
        train_seq = input_data[i:i+tw_in]
        train_label = input_data[i+tw_in:i+tw_in+tw_out,2] # The third column is the target
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# Create training and test data
train_inout_seq = create_inout_sequences(train, input_seq_len, output_seq_len)
test_inout_seq = create_inout_sequences(test, input_seq_len, output_seq_len)
print("train_inout_seq: ", train_inout_seq[0])

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        
        return h, c

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_seq, h, c):
        # input_seq(batch_size, input_size)
        input_seq = input_seq.unsqueeze(1)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, seq_len, num * hidden_size)
        pred = self.linear(output.squeeze(1))  # pred(batch_size, 1, output_size)

        return pred, h, c


# Define seq2seq model
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Encoder = Encoder(input_size, hidden_size, num_layers, batch_size)
        self.Decoder = Decoder(input_size, hidden_size, num_layers, output_size, batch_size)

    def forward(self, input_seq):
        target_len = self.output_size 
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        outputs = torch.zeros(batch_size, self.input_size, self.output_size).to(device)
        decoder_input = input_seq[:, -1, :]
        for t in range(target_len):
            decoder_output, h, c = self.Decoder(decoder_input, h, c)
            outputs[:, :, t] = decoder_output
            decoder_input = decoder_output

        return outputs[:, 0, :]


# Initialize model, loss function and optimizer
model = Seq2Seq(input_size=3, hidden_size=hidden_size, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train model
for i in range(epoch_num):
    start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        seq = Variable(torch.from_numpy(seq)).float()
        labels = Variable(torch.from_numpy(labels)).float()
        y_pred, _ = model(seq, None)
        single_loss = criterion(y_pred, labels)
        single_loss.backward()
        optimizer.step()
        epoch_loss += single_loss.item()
        num_batches += 1
    end_time = time.time()
    avg_loss = epoch_loss / num_batches
    print(f'epoch: {i:3} loss: {avg_loss:10.8f} time: {end_time-start_time:5}s')
    if avg_loss < early_stop_loss:
        break

# Test model
model = model.eval()
test_inputs = test[:input_seq_len]
for i in range(len(test) - input_seq_len):
    seq = torch.FloatTensor(test_inputs[-input_seq_len:])
    with torch.no_grad():
        test_inputs = np.append(test_inputs, model(seq, None)[0].item())

# Calculate MSE
mse = mean_squared_error(test[input_seq_len:], test_inputs[input_seq_len:])
print('The MSE of the model on the test set is', mse)

# Plot actual and predicted values
plt.figure(figsize=(14,5))
plt.plot(range(len(data)), scaler.inverse_transform(data[:,2].reshape(-1,1)), color='blue')
plt.plot(range(len(data)-len(test_inputs), len(data)), scaler.inverse_transform(test_inputs.reshape(-1,1)), color='red')
plt.show()
plt.savefig("seq2seq.png")