import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

# Define the sequence lengths and other variables
input_seq_len = 180
output_seq_len = 180
hidden_layer_size = 64
num_layers = 2
early_stop_loss = 0.002
early_stop_diff = 0.0005
epochs = 10
lr = 0.002

# Load the data
data = pd.read_csv('VM500.csv', usecols=[0,1,2])

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data)

# Split the data into training and test sets
train_size = int(len(data_normalized) * 0.7)
train_set = data_normalized[:train_size]
test_set = data_normalized[train_size:]
test_set_origin = data[train_size:]
test_set_origin = np.array(test_set_origin)

# Convert the data into tensors
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

# Create sequences of input_seq_len time steps for training
def create_inout_sequences(input_data, tw_in, tw_out):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw_in-tw_out):
        train_seq = input_data[i:i+tw_in]
        train_label = input_data[i+tw_in:i+tw_in+tw_out,2] # The third column is the target
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_set, input_seq_len, output_seq_len)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size),
                            torch.zeros(num_layers,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-output_seq_len:]

# Train the model
model = LSTM(input_size=3, hidden_layer_size=hidden_layer_size, output_size=1, num_layers=num_layers)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

last_loss = 1
count = 0
for i in range(epochs):
    start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size),
                        torch.zeros(num_layers, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels.unsqueeze(1))
        single_loss.backward()
        optimizer.step()

        epoch_loss += single_loss.item()
        num_batches += 1

    end_time = time.time()
    avg_loss = epoch_loss / num_batches
    print(f'epoch: {i:3} loss: {avg_loss:10.8f} time: {end_time-start_time:5}s')
    if avg_loss < early_stop_loss:
        break
    if abs(avg_loss - last_loss) < early_stop_diff:
        count = count + 1
        if count > 2:
            break
    else:
        count = 0
    last_loss = avg_loss


# Make predictions on the test set
test_outputs = []
model.eval()

for i in range(0, len(test_set)-input_seq_len-output_seq_len, input_seq_len):
    seq = test_set[i:i+input_seq_len, 0:3]
    with torch.no_grad():
        model.hidden = (torch.zeros(num_layers, 1, model.hidden_layer_size),
                        torch.zeros(num_layers, 1, model.hidden_layer_size))
        pre = model(seq)
        if i == input_seq_len:
            print("model(seq)", pre.tolist())
        test_outputs.extend(pre.tolist())

#print("test dataset output:", test_outputs)
# Calculate the MSE error
pred_res = test_outputs[input_seq_len:]
pred_res = np.array(pred_res)

full_pred_res = np.zeros((len(pred_res), 3))
full_pred_res[:, 2] = pred_res.flatten()
inverse_data = scaler.inverse_transform(full_pred_res)
pred_res_origin = inverse_data[:, 2]

#actual_predictions = scaler.inverse_transform(np.array(test_outputs[input_seq_len:] ).reshape(0, 1))
x = np.array(range(input_seq_len, len(test_outputs)+input_seq_len))

mse = mean_squared_error(test_set[x,2], test_outputs)
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Plot the actual vs predicted values
plt.grid(True)
print(test_set_origin)
#print(pred_res_origin)
plt.plot(test_set_origin[:,2], label='Real', color='r')
plt.plot(pred_res_origin, label='Predicted', color='b')
plt.show()
plt.savefig("lstm1.png")