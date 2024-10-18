import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

# Define the sequence lengths and other variables
predict_step = 30
input_seq_len = 180
output_seq_len = 180
hidden_layer_size = 64
num_layers = 2
early_stop_loss = 0.002
early_stop_diff = 0.0001
epochs = 10
lr = 0.005

# Load the data
data = pd.read_csv('VM.csv', usecols=[0,1])
model_mse_path = "model_mse.pth"
model_skewed_path = "model_skewed.pth"

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
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
    for i in range(0, L-tw_in-tw_out):
        train_seq = input_data[i:i+tw_in]
        train_label = input_data[i+tw_in:i+tw_in+tw_out,1] # The third column is the target
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


class AsymmetricLoss(nn.Module):
    def __init__(self, high_weight=1.0, low_weight=4.0):
        super(AsymmetricLoss, self).__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight

    def forward(self, y_pred, y_true):
        loss = torch.where(y_pred > y_true, 
                           self.high_weight * (y_pred - y_true) ** 2,  # 高估时权重更大
                           self.low_weight * (y_true - y_pred) ** 2)   # 低估时权重较小
        return loss.mean()
    
# Train the model

def train_model(loss_function):
    model = LSTM(input_size=2, hidden_layer_size=hidden_layer_size, output_size=1, num_layers=num_layers)
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
        print(f'epoch: {i:3} loss: {avg_loss:10.8f} loss_function: {loss_function} time: {end_time-start_time:5}s')
        if avg_loss < early_stop_loss:
            break
        if abs(avg_loss - last_loss) < early_stop_diff:
            count = count + 1
            if count > 2:
                break
        else:
            count = 0
        last_loss = avg_loss
    return model

model = train_model(loss_function = nn.MSELoss())
model_skewed = train_model(loss_function = AsymmetricLoss(high_weight=0.333, low_weight=3.0))

# Make predictions on the test set
test_outputs = []
test_outputs_skewed = []
model.eval()
model_skewed.eval()

for i in range(0, len(test_set)-input_seq_len-output_seq_len, predict_step):
    seq = test_set[i:i+input_seq_len, 0:2]
    duration = 0
    with torch.no_grad():
        model.hidden = (torch.zeros(num_layers, 1, model.hidden_layer_size),
                        torch.zeros(num_layers, 1, model.hidden_layer_size))
        model_skewed.hidden = (torch.zeros(num_layers, 1, model.hidden_layer_size),
                        torch.zeros(num_layers, 1, model.hidden_layer_size))
        pre = model(seq)
        start_time = time.time()
        pre_skewed = model_skewed(seq)
        end_time = time.time()
        duration += end_time() - start_time()
        test_outputs.extend(pre.tolist()[:predict_step])
        test_outputs_skewed.extend(pre_skewed.tolist()[:predict_step])

#print("test dataset output:", test_outputs)
# Calculate the MSE error
#pred_res = test_outputs[input_seq_len:]
pred_res = test_outputs
pred_res = np.array(pred_res)
pred_res_skewed = test_outputs_skewed
pred_res_skewed = np.array(pred_res_skewed)

real_res = test_set_origin[:,1]
real_res_print = real_res[input_seq_len:len(real_res)-153]

full_pred_res = np.zeros((len(pred_res), 2))
full_pred_res[:, 1] = pred_res.flatten()
inverse_data = scaler.inverse_transform(full_pred_res)
pred_res_origin = inverse_data[:, 1]

full_pred_res_skewed = np.zeros((len(pred_res_skewed), 2))
full_pred_res_skewed[:, 1] = pred_res_skewed.flatten()
inverse_data_skewed = scaler.inverse_transform(full_pred_res_skewed)
pred_res_origin_skewed = inverse_data_skewed[:, 1]

print("length of test_outputs:{}".format(len(real_res_print)))
print("length of pred_res:{}".format(len(pred_res_origin)))

#actual_predictions = scaler.inverse_transform(np.array(test_outputs[input_seq_len:] ).reshape(0, 1))
x = np.array(range(input_seq_len, len(test_outputs)+input_seq_len))

mse = mean_squared_error(test_set[x,1], test_outputs)
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

def average_error(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("length is not equal.")
    
    errors = [abs(list1[i] - list2[i]) / list1[i] for i in range(len(list1))]
    return sum(errors) / len(errors)

def count_long_intervals(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("length is not equal.")
    
    count = 0
    in_interval = False
    interval_length = 0

    for i in range(len(list1)):
        if list2[i] < list1[i]:
            if not in_interval:
                in_interval = True
                interval_length = 1  # 开始新的区间
            else:
                interval_length += 1  # 增加当前区间的长度
        else:
            if in_interval:
                if interval_length > 60:
                    count += 1  # 结束区间且长度大于60
                in_interval = False
                interval_length = 0  # 重置区间长度

    # 检查最后一个区间
    if in_interval and interval_length > 60:
        count += 1

    return count

def count_large_intervals(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("length is not equal.")
    
    count = 0
    in_interval = False
    interval_length = 0

    for i in range(len(list1)):
        if list2[i] < list1[i] - 50:  # 判断是否小于100
            if not in_interval:
                in_interval = True
                interval_length = 1  # 开始新的区间
            else:
                interval_length += 1  # 增加当前区间的长度
        else:
            if in_interval:
                if interval_length > 10:
                    count += 1  # 结束区间且长度大于10
                in_interval = False
                interval_length = 0  # 重置区间长度

    # 检查最后一个区间
    if in_interval and interval_length > 10:
        count += 1

    return count


print('diff of real and predict_mse is {}'.format(average_error(real_res_print, pred_res_origin)))
print('diff of real and predict_skewed is {}'.format(average_error(real_res_print, pred_res_origin_skewed)))
print('sla predict_mse is {}'.format(count_long_intervals(real_res_print, pred_res_origin) + count_large_intervals(real_res_print, pred_res_origin)))
print('sla predict_skewed is {}'.format(count_long_intervals(real_res_print, pred_res_origin) + count_large_intervals(real_res_print, pred_res_origin)))


# Plot the actual vs predicted values
plt.grid(True)
#print(test_set_origin)
#print(pred_res_origin)
plt.plot(real_res_print, label='Real', color='r')
plt.plot(pred_res_origin, label='Predicted_MSE', color='b')
plt.plot(pred_res_origin_skewed, label='Predicted_Skewed', color='g')
plt.legend()
plt.show()
plt.savefig("lstm1.png")

import json

def write_list_to_file(my_list, filename):
    with open(filename, 'w') as file:
        json.dump(my_list, file)

def read_list_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

file1 = "mse.json"
file2 = "skewed.json"

write_list_to_file(pred_res_origin, file1)
write_list_to_file(pred_res_origin_skewed, file2)
torch.save(model.state_dict(), model_mse_path)
torch.save(model_skewed.state_dict(), model_skewed_path)