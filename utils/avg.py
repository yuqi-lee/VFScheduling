import pandas as pd

# 读取数据文件
data = pd.read_csv('VM.csv', header=None, names=['timestamp', 'bandwidth'])

# 初始化新的列表用于存储结果
new_data = []

# 处理每三个连续的数据
for i in range(0, len(data) - 2, 3):
    timestamp = data['timestamp'].iloc[i]  # 取第一个数据的时间戳
    avg_bandwidth = data['bandwidth'].iloc[i:i+3].mean()  # 计算三个数据的平均值
    new_data.append([timestamp, avg_bandwidth])

# 将结果转换为DataFrame并保存到新文件
new_df = pd.DataFrame(new_data, columns=['timestamp', 'bandwidth_avg'])
new_df.to_csv('VM3.csv', index=False, header=False)
