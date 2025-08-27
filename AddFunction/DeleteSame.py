# 定义要处理的文件名
input_file = 'C:\\Users\\19551\\Desktop\\Domain\\MDF-Net-master\\data\\BUS_A\\train.txt'
output_file = 'C:\\Users\\19551\\Desktop\\Domain\\MDF-Net-master\\data\\BUS_A\\train1.txt'

# 使用列表来存储唯一的行
unique_lines = []
seen = set()

# 读取文件并存储唯一行
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        stripped_line = line.strip()
        if stripped_line not in seen:
            seen.add(stripped_line)
            unique_lines.append(stripped_line)

# 将唯一的行写入新的文件
with open(output_file, 'w', encoding='utf-8') as file:
    for line in unique_lines:
        file.write(line + '\n')

print(f"处理完成，唯一行已写入 {output_file}。")