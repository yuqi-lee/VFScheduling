import json
import matplotlib.pyplot as plt
import random

plt.rcParams['font.size'] = 19


#plt.figure(figsize=(16, 6))

file1 = f"real.json"
file2 = f"mse.json"
file3 = f"skewed.json"
file4 = f"mbfq.json"
file5 = f"mbfq_opt.json"

# 读取JSON文件并返回list
def read_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

# 读取三个文件
data1 = read_json(file1)
data2 = read_json(file2)
data3 = read_json(file3)
data4 = read_json(file4)
data5 = read_json(file5)


# 创建x轴数据
l = min(len(data1), len(data2), len(data3), len(data4))
x_values = range(l-3)

# 绘制曲线
#plt.ylim(0, 2500)

plt.figure(figsize=(16, 6))

plt.plot(x_values, data1[4:l+1], label='Real Bandwidth', color='r', linewidth=2)
#plt.plot(x_values, data2[:l], label='Predicted_MSE', color='b')
#plt.plot(x_values, data3[:l], label='Predicted_Skewed', color='g')
#plt.plot(x_values, data4[3:l], label='Allocated Bandwidth', linewidth=2)
plt.plot(x_values, data5[3:l], label='Allocated Bandwidth', color='g', linewidth=2)

# 添加图例和标签
plt.legend(loc='upper right')
plt.xlabel('Time(s)')
plt.ylabel('Bandwidth(KB/s)')

# 显示图形
plt.show()
plt.savefig("lstm.png")
plt.savefig("lstm.pdf")

def average_error(list1, list2): 
    errors = [abs(list1[i] - list2[i]) / list1[i] for i in range(min(len(list1), len(list2)))]
    return sum(errors) / len(errors)

def count_long_intervals(list1, list2): 
    count = 0
    in_interval = False
    interval_length = 0

    for i in range(min(len(list1), len(list2))):
        if list2[i] < list1[i]:
            if not in_interval:
                in_interval = True
                interval_length = 1  # 开始新的区间
            else:
                interval_length += 1  # 增加当前区间的长度
        else:
            if in_interval:
                if interval_length >= 60:
                    count += 1  # 结束区间且长度大于60
                in_interval = False
                interval_length = 0  # 重置区间长度

    # 检查最后一个区间
    if in_interval and interval_length > 60:
        count += 1

    return count

def count_large_intervals(list1, list2):
    count = 0
    in_interval = False
    interval_length = 0

    for i in range(min(len(list1), len(list2))):
        if list2[i] < list1[i] - 200:  
            if not in_interval:
                in_interval = True
                interval_length = 1  # 开始新的区间
            else:
                interval_length += 1  # 增加当前区间的长度
        else:
            if in_interval:
                if interval_length >= 10:
                    count += 1  # 结束区间且长度大于10
                in_interval = False
                interval_length = 0  # 重置区间长度

    # 检查最后一个区间
    if in_interval and interval_length > 10:
        count += 1

    return count


print('diff of real and predict_mse is {}'.format(average_error(data1, data2)))
print('diff of real and predict_skewed is {}'.format(average_error(data1, data3)))
print('sla predict_mse is {}'.format(count_long_intervals(data1, data2) + count_large_intervals(data1, data2)))
print('sla predict_skewed is {}'.format(count_long_intervals(data1, data3) + count_large_intervals(data1, data3)))
