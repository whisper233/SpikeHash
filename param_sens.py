
# ("0.01" "0.05" "0.1" "0.5" "1" "5")

flickr = list()
flickr.append([[0.860, 0.871], [0.866, 0.879], [0.868, 0.891], [0.876, 0.905], [0.888, 0.911], [0.886, 0.910]]) # alpha
flickr.append([[0.880, 0.903], [0.881, 0.907], [0.882, 0.908], [0.888, 0.911], [0.887, 0.907], [0.891, 0.909]]) # beta
flickr.append([[0.884, 0.896], [0.887, 0.904], [0.888, 0.911], [0.888, 0.911], [0.884, 0.911], [0.882, 0.903]]) # eta

x_labels = [0.01, 0.05, 0.1, 0.5, 1, 5]  # 自定义的横坐标标签
x_positions = range(len(x_labels))      # 均匀分布的横坐标位置

subfig_names = [r'$\alpha$', r'$\beta$', r'$\eta$']

data = flickr

import matplotlib.pyplot as plt

# 预定义数据：3 个子图，每个子图有两条折线的数据
# data = [
#     ([(0, 1), (1, 3), (2, 2), (3, 5)], [(0, 2), (1, 4), (2, 3), (3, 6)]),  # 子图 1 的两条线
#     ([(0, 3), (1, 2), (2, 4), (3, 1)], [(0, 1), (1, 5), (2, 2), (3, 4)]),  # 子图 2 的两条线
#     ([(0, 4), (1, 1), (2, 3), (3, 2)], [(0, 2), (1, 3), (2, 5), (3, 1)])   # 子图 3 的两条线
# ]

# 创建一个包含 3 个子图的大图 (1x3 布局)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 行 3 列，调整 figsize 以适应宽度

# 遍历每个子图并绘制折线图
for i, ax in enumerate(axes):
    lines_data = data[i]  # 获取当前子图的两条线的数据
    # x1, y1 = zip(*line1_data)  # 解压第一条线的 x 和 y 数据
    # x2, y2 = zip(*line2_data)  # 解压第二条线的 x 和 y 数据

    y1 = [i[0] for i in lines_data]
    y2 = [i[1] for i in lines_data]

    ax.plot(x_positions, y1, label='I2T', color='blue', marker='o')  # 绘制第一条线
    ax.plot(x_positions, y2, label='T2I', color='orange', marker='s')  # 绘制第二条线
    # ax.set_title(f'Subplot {i+1}')  # 设置子图标题
    ax.legend(fontsize=14, loc='lower right')  # 添加图例
    ax.set_ylim(0.5, 1)

    ax.set_xticks(x_positions)  # 设置刻度位置为均匀分布
    ax.set_xticklabels(x_labels, fontsize=14)  # 设置刻度标签为自定义的 x_labels
    ax.tick_params(axis='y', labelsize=14)     # 设置纵坐标字体大小

    ax.text(0.5, -0.2, subfig_names[i], fontsize=24,
            ha='center', va='center', transform=ax.transAxes)
    ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')

# 调整布局以避免重叠
plt.tight_layout()

# 显示图像
plt.show()
plt.savefig('output.png')

