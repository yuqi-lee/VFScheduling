import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

torch.set_num_threads(16)

file_path = '/users/YuqiLi/fastStorage/2013-8/500.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path, sep=';')
data.columns = data.columns.str.strip()

# 2. 选择需要的列作为特征
features = ['CPU usage [MHZ]', 'Memory usage [KB]', 'Disk read throughput [KB/s]', 
            'Disk write throughput [KB/s]', 'Network received throughput [KB/s]', 
            'Network transmitted throughput [KB/s]']
target = 'Network transmitted throughput [KB/s]'

# 3. 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# 4. 设置模型参数
seq_length = 180          # 序列长度（使用多少个样本来预测未来值）
hidden_size = 64          # LSTM 隐藏层大小
num_layers = 2            # LSTM 层数
output_size = 1           # 输出大小
learning_rate = 0.01     # 学习率
num_epochs = 10          # 训练轮数
train_split = 0.7         # 训练集比例
patience = 10             # 提前停止的耐心值，当损失不下降时等待的最大轮数


train_size = int(len(scaled_data) * train_split)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 6. 生成输入数据（使用前180个样本来预测最后一个样本）
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length, :]
        y = data[i + seq_length, -1]  # Network transmitted throughput
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 生成训练和测试的序列
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 调整为2D
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)    # 调整为2D

# 7. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)  # 初始化隐藏层
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)  # 初始化细胞状态
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # 只取LSTM最后一个输出
        return out

# 8. 提前停止类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 初始化模型、损失函数和优化器
input_size = X_train.shape[2]
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 9. 训练模型并引入提前停止
early_stopping = EarlyStopping(patience=patience, verbose=True)

model.train()
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    # 记录训练集损失
    train_losses.append(loss.item())

    # 在测试集上计算损失
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = loss_fn(test_outputs, y_test).item()
        test_losses.append(test_loss)

    # 打印每个epoch的训练和验证损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}')
    
    # 检查是否提前停止
    early_stopping(test_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

# 10. 在测试集上验证模型
model.eval()
with torch.no_grad():
    predicted = model(X_test).cpu().numpy()
    y_test_np = y_test.cpu().numpy()

# 11. 反向缩放结果
predicted = scaler.inverse_transform(np.concatenate([np.zeros((predicted.shape[0], 5)), predicted], axis=1))[:, -1]
y_test_np = scaler.inverse_transform(np.concatenate([np.zeros((y_test_np.shape[0], 5)), y_test_np.reshape(-1, 1)], axis=1))[:, -1]

# 12. 计算均方误差
mse = mean_squared_error(y_test_np, predicted)
print(f'Test MSE: {mse:.4f}')

# 13. 可视化结果
plt.plot(y_test_np, label='True Network transmitted throughput')
plt.plot(predicted, label='Predicted Network transmitted throughput')
plt.legend()
plt.show()

# 14. 可视化训练和验证损失
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()
