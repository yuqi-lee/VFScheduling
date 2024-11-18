import json
import torch
import torch.nn as nn

file1 = f"real.json"
file2 = f"mse.json"
file3 = f"skewed.json"
model_path = "model_mse.pth"

def read_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data
def write_list_to_file(my_list, filename):
    with open(filename, 'w') as file:
        json.dump(my_list, file)

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
        return predictions[-output_seq_len:]


AR = 500  # allocated rate
TR = 0  # target rate
RU = 0  # 迭代次数
W = 1  # weight
NR = 0  # 新分配的速率
BelowTR = False
Below_85_Percent_AR = 0  # 连续低于85%的次数
NETWORK_CAPACITY = 300000
Guaranteed_BW = 1000

count_1 = 0
count_2 = 0

real_data = read_json(file1)
pred_data = read_json(file3)

l = min(len(real_data), len(pred_data))

def mbfq(SR):
    global AR  
    global TR  
    global RU  
    global W
    global NR
    global BelowTR
    global Below_85_Percent_AR
    global NETWORK_CAPACITY
    global Guaranteed_BW
    global count_1
    global count_2

    w_all = 0
    vm_count_all = 0
    avail_bw_all = NETWORK_CAPACITY
    

    if SR < 0.85 * AR:
        TR = 1.1 * SR
        RU = max(0, RU - 1)
        count_1 += 1
    elif SR > 0.95 * AR:
        RU = min(3, RU + 1)
        alpha = 0.0
        count_2 += 1
        if RU == 1:
            alpha = 1.2
        elif RU == 2:
            alpha = 1.5
        elif RU == 3:
            alpha = 2.0
        TR = min(alpha * AR, AR + 0.1 * NETWORK_CAPACITY)
    else:
        TR = AR
        RU = max(0, RU - 1)
    #TR = (2*TR + pred) / 3
    NR=TR
    AR = NR
    print(AR, SR, NR, TR)
    return NR

def mbfq_opt(SR, pred):
    global AR  
    global TR  
    global RU  
    global W
    global NR
    global BelowTR
    global Below_85_Percent_AR
    global NETWORK_CAPACITY
    global Guaranteed_BW
    global count_1
    global count_2

    w_all = 0
    vm_count_all = 0
    avail_bw_all = NETWORK_CAPACITY
    
    sum_pred = 0.0
    max_pred = 0.0
    for i in range(len(pred)):
        sum_pred += pred[i]
        max_pred = max(max_pred, pred[i])
    avg_pred = sum_pred / len(pred)
    first_avg_pred = sum(pred[:10]) / len(pred[:10])
    first_max_pred = max(pred[:10])

    if SR < 0.85 * AR:
        if first_max_pred < 0.9*AR:
            TR = 1.1 * SR
            count_1 += 1
        else:
            TR = AR
        RU = max(0, RU - 1)
    elif SR > 0.95 * AR and first_max_pred > AR: #扩容
        RU = min(3, RU + 1)
        count_2 += 1
        alpha = 0.0
        if RU == 1:
            alpha = 1.2 if avg_pred > 1.1 * AR  else 1.1
        elif RU == 2:
            alpha = 1.5 if avg_pred > 1.2 * AR  else 1.2
        elif RU == 3:
            alpha = 1.8 if avg_pred > 1.5 * AR  else 1.5
        TR = alpha * AR
    else:
        TR = AR
        RU = max(0, RU - 1)

    NR=TR
    AR = NR
    print(AR, SR, NR, TR)
    return NR

res = []
#model = LSTM(input_size=2, hidden_layer_size=hidden_layer_size, output_size=1, num_layers=num_layers).to(device)
#model.load_state_dict(torch.load(model_skewed_path))
#model.eval()

for i in range(l):
    pred = pred_data[i : i+30]
    if i + 1 < len(pred_data):
        p = pred_data[i+1]
    else:
        p = real_data[i]
    b = mbfq_opt(real_data[i], pred)
    #b = mbfq(real_data[i])
    res.append(b)

print("count_1 = {}".format(count_1))
print("count_2 = {}".format(count_2))

write_list_to_file(res, "mbfq_opt.json")
