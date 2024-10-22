import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_dtype(torch.float64)

# Define the sequence lengths and other variables
predict_step = 6
input_seq_len = 48
output_seq_len = predict_step
hidden_layer_size = 64
num_layers = 2
early_stop_loss_mse = 0.003
early_stop_loss_skewed = 0.004
early_stop_diff = 0.0001
lr = 0.01

# Load the data
data = pd.read_csv('VM3.csv', usecols=[0,1])
model_mse_path = "model_mse.pth"
model_skewed_path = "model_skewed.pth"

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Split the data into training and test sets
train_size = 2130
start = 2130
end = 2400
train_set = data_normalized
test_set = data_normalized[start:]
test_set_origin = np.array(data[start:])

# Convert the data into tensors
train_set = torch.FloatTensor(train_set).to(device)
test_set = torch.FloatTensor(test_set).to(device)

# Create sequences of input_seq_len time steps for training
def create_inout_sequences(input_data, tw_in, tw_out):
    inout_seq = []
    L = len(input_data)
    for i in range(0, L-tw_in-tw_out):
        train_seq = input_data[i:i+tw_in]
        train_label = max(input_data[i+tw_in:i+tw_in+tw_out, 1])
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_set, input_seq_len, output_seq_len)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size).to(device),
                            torch.zeros(num_layers, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return max(predictions[-output_seq_len:])

class AsymmetricLoss(nn.Module):
    def __init__(self, low_weight=1.0, medium_weight=4.0, high_weight=8.0):
        super(AsymmetricLoss, self).__init__()
        self.low_weight = low_weight
        self.medium_weight = medium_weight
        self.high_weight = high_weight

    def forward(self, y_pred, y_true):
        loss = torch.where(y_pred > y_true, 
                           (y_pred - y_true) ** 2,
                           torch.where(y_true - y_pred < 0.02,
                                        2 * (y_pred - y_true) ** 2,
                                        torch.where(y_true - y_pred > 0.1,
                                                    300 * (y_true - y_pred) * (y_pred - y_true) ** 2,
                                                    160 * (y_true - y_pred) * (y_pred - y_true) ** 2)))
        return loss.mean()

# Train the model
def train_model(loss_function, early_stop_loss, epochs):
    model = LSTM(input_size=2, hidden_layer_size=hidden_layer_size, output_size=1, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = float('inf')
    count = 0
    for i in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                                 torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq.to(device))

            single_loss = loss_function(y_pred, labels.unsqueeze(1).to(device))
            single_loss.backward()
            optimizer.step()

            epoch_loss += single_loss.item()
            num_batches += 1

        end_time = time.time()
        avg_loss = epoch_loss / num_batches
        print(f'epoch: {i:3} loss: {avg_loss:10.8f} loss_function: {loss_function} time: {end_time-start_time:5}s')
        if avg_loss < early_stop_loss or abs(avg_loss - last_loss) < early_stop_diff:
            count += 1
            if count > 3:
                break
        else:
            count = 0
        last_loss = avg_loss
    return model

model = train_model(loss_function=nn.MSELoss(), early_stop_loss=early_stop_loss_mse, epochs=10)
model_skewed = train_model(loss_function=AsymmetricLoss(), early_stop_loss=early_stop_loss_skewed, epochs=10)

#model = LSTM(input_size=2, hidden_layer_size=hidden_layer_size, output_size=1, num_layers=num_layers).to(device)
#model.load_state_dict(torch.load(model_mse_path))

#model_skewed = LSTM(input_size=2, hidden_layer_size=hidden_layer_size, output_size=1, num_layers=num_layers).to(device)
#model_skewed.load_state_dict(torch.load(model_skewed_path))

# Make predictions on the test set
test_outputs = []
test_outputs_skewed = []
model.eval()
model_skewed.eval()

for i in range(0, len(test_set) - input_seq_len - output_seq_len, predict_step):
    seq = test_set[i:i + input_seq_len, 0:2]
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                             torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))
        model_skewed.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).to(device),
                                     torch.zeros(num_layers, 1, model.hidden_layer_size).to(device))
        pre = model(seq.to(device))
        pre_skewed = model_skewed(seq.to(device))
        res_pre = [max(pre.tolist()[:predict_step])] * predict_step
        res_pre_skewed = [max(pre_skewed.tolist()[:predict_step])] * predict_step
        test_outputs.extend(res_pre)
        test_outputs_skewed.extend(res_pre_skewed)

pred_res = np.array(test_outputs)
pred_res_skewed = np.array(test_outputs_skewed)
real_res = test_set_origin[input_seq_len:, 1]

# Inverse transform predictions
full_pred_res = np.zeros((len(pred_res), 2))
full_pred_res[:, 1] = pred_res.flatten()
inverse_data = scaler.inverse_transform(full_pred_res)
pred_res_origin = inverse_data[:, 1]

full_pred_res_skewed = np.zeros((len(pred_res_skewed), 2))
full_pred_res_skewed[:, 1] = pred_res_skewed.flatten()
inverse_data_skewed = scaler.inverse_transform(full_pred_res_skewed)
pred_res_origin_skewed = inverse_data_skewed[:, 1]

# Calculate metrics
#mse = mean_squared_error(real_res, pred_res_origin)
#print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

def average_error(list1, list2): 
    errors = [abs(list1[i] - list2[i]) / list1[i] for i in range(min(len(list1), len(list2)))]
    return sum(errors) / len(errors)

print('diff of real and predict_mse is {}'.format(average_error(real_res, pred_res_origin)))
print('diff of real and predict_skewed is {}'.format(average_error(real_res, pred_res_origin_skewed)))

# Plot the actual vs predicted values
plt.grid(True)
plt.plot(real_res, label='Real', color='r')
plt.plot(pred_res_origin, label='Predicted_MSE', color='b')
plt.plot(pred_res_origin_skewed, label='Predicted_Skewed', color='g')
plt.legend()
plt.show()
plt.savefig("lstm1.png")
plt.savefig("lstm1.pdf")

# Save predictions to JSON
def write_list_to_file(my_list, filename):
    with open(filename, 'w') as file:
        json.dump(my_list, file)

write_list_to_file(pred_res_origin.tolist(), "mse.json")
write_list_to_file(pred_res_origin_skewed.tolist(), "skewed.json")
write_list_to_file(real_res.tolist(), "real.json")

torch.save(model.state_dict(), model_mse_path)
torch.save(model_skewed.state_dict(), model_skewed_path)
