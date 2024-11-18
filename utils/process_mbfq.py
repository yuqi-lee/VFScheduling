# 读取原始文件并提取第三列
with open('VM3.csv', 'r') as infile, open('mbfq.csv', 'w') as outfile:
    for line in infile:
        # 去掉空白字符并分割每行数据
        columns = line.strip().split(',')
        # 检查是否至少有三列数据
        if len(columns) >= 2:
            # 提取第三列数据
            third_column = columns[1]
            # 将第三列数据写入到新文件中
            outfile.write(third_column + '\n')