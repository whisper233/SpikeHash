import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
from tqdm import tqdm
from matplotlib import colormaps
import matplotlib.lines as mlines
from tqdm import tqdm

# plt.rc('font', family='Times New Roman')

# label_names = [
# 'sky'
# , 'clouds'
# , 'person'
# , 'water'
# , 'animal'
# , 'grass'
# , 'buildings'
# , 'window'
# , 'plants'
# , 'lake'
# , 'ocean'
# , 'road'
# , 'flowers'
# , 'sunset'
# , 'reflection'
# , 'rocks'
# , 'vehicle'
# , 'snow'
# , 'tree'
# , 'beach'
# , 'mountain']


# def filter_samples(te_BI, te_BT, te_L, top_k=10, samples_per_class=50):
#     # 统计每个标签的样本数
#     label_counts = np.sum(te_L, axis=0)
    
#     # 获取top10标签（按样本数降序）
#     top_labels = np.argsort(-label_counts)[5:top_k+5]
#     top_labels = sorted(top_labels, key=lambda x: -label_counts[x])
    
#     selected_indices = []
#     used_indices = set()
    
#     for current_label in top_labels:
#         # 获取其他top标签
#         other_labels = [l for l in top_labels if l != current_label]
        
#         # 获取当前标签的所有样本索引
#         label_indices = np.where(te_L[:, current_label] == 1)[0]
        
#         # 划分纯净样本和非纯净样本
#         pure_samples = []
#         mixed_samples = []
        
#         for idx in label_indices:
#             if idx in used_indices:
#                 continue
#             if np.any(te_L[idx, other_labels]):
#                 mixed_samples.append(idx)
#             else:
#                 pure_samples.append(idx)
        
#         # 优先选择纯净样本
#         selected = []
#         remain = samples_per_class
        
#         # 第一阶段选择
#         if len(pure_samples) >= remain:
#             selected = np.random.choice(pure_samples, remain, replace=False)
#         else:
#             selected = pure_samples.copy()
#             remain -= len(selected)
            
#             # 第二阶段选择
#             if len(mixed_samples) >= remain:
#                 selected += np.random.choice(mixed_samples, remain, replace=False).tolist()
#             else:
#                 selected += mixed_samples[:remain]
        
#         # 更新已选索引
#         used_indices.update(selected)
#         selected_indices.extend(selected)
        
#     # 转换为numpy数组
#     selected_indices = np.array(selected_indices)
    
#     # 验证结果
#     # assert len(selected_indices) == top_k * samples_per_class, \
#     #     f"筛选数量不符预期，实际筛选到{len(selected_indices)}个样本"
    
#     return (
#         te_BI[selected_indices], 
#         te_BT[selected_indices], 
#         te_L[selected_indices]
#     )

# 使用示例
# te_BI_filtered, te_BT_filtered, te_L_filtered = filter_samples(te_BI, te_BT, te_L)


def filter_top_labels_and_samples(te_BI, te_BT, te_L, top_n=10, samples_per_label=50):
    """
    根据标签数量筛选出 top_n 个样本数量最多的标签，并为每个标签筛选出 samples_per_label 个样本。
    
    参数:
        te_BI (np.ndarray): 图像特征矩阵，形状为 (n_samples, n_features_i)
        te_BT (np.ndarray): 文本特征矩阵，形状为 (n_samples, n_features_t)
        te_L (np.ndarray): one-hot 类型的标签矩阵，形状为 (n_samples, n_classes)
        top_n (int): 筛选的 top_n 个标签
        samples_per_label (int): 每个标签需要的样本数
    
    返回:
        filtered_te_BI (np.ndarray): 筛选后的图像特征矩阵
        filtered_te_BT (np.ndarray): 筛选后的文本特征矩阵
        filtered_te_L (np.ndarray): 筛选后的标签矩阵（仅保留主要标签）
    """
    # Step 1: 统计每个标签的样本数量
    label_counts = np.sum(te_L, axis=0)  # 每个标签的样本总数
    top_labels = np.argsort(label_counts)[-(top_n+3):-3]  # 获取样本数量最多的 top_n 个标签索引

    # Step 2: 初始化结果列表
    filtered_indices = []  # 用于存储筛选出的样本索引

    # Step 3: 遍历每个 top 标签并筛选样本
    for label in top_labels:
        # 找到当前标签的所有样本索引
        label_sample_indices = np.where(te_L[:, label] == 1)[0]

        # 筛选出优先不包含其他 top 标签的样本
        pure_samples = []
        mixed_samples = []

        for idx in label_sample_indices:
            other_top_labels = np.sum(te_L[idx, top_labels])  # 当前样本是否包含其他 top 标签
            if other_top_labels == 1:  # 只有当前标签，没有其他 top 标签
                pure_samples.append(idx)
            else:  # 包含其他 top 标签
                mixed_samples.append(idx)

        # 优先选择纯样本
        selected_indices = pure_samples[:samples_per_label]

        # 如果纯样本不足，则补充混合样本
        if len(selected_indices) < samples_per_label:
            needed = samples_per_label - len(selected_indices)
            selected_indices += mixed_samples[:needed]

        # 添加到最终筛选索引中
        filtered_indices.extend(selected_indices[:samples_per_label])

    # Step 4: 根据筛选出的索引提取对应的样本
    filtered_te_BI = te_BI[filtered_indices]
    filtered_te_BT = te_BT[filtered_indices]
    filtered_te_L = te_L[filtered_indices]

    # Step 5: 修改标签矩阵，只保留主要标签
    final_te_L = np.zeros_like(filtered_te_L)  # 创建一个全零矩阵
    for i, idx in enumerate(filtered_indices):
        active_labels = np.where(te_L[idx] == 1)[0]  # 获取当前样本的所有激活标签
        common_labels = set(active_labels).intersection(top_labels)  # 找到与 top 标签的交集
        if common_labels:
            main_label = next(iter(common_labels))  # 主要标签
            final_te_L[i, main_label] = 1  # 仅保留主要标签

    return filtered_te_BI, filtered_te_BT, final_te_L


def show_t_sne(dataset, file_name):
    file_path = os.path.join('/data/zhangzhen/logs/resnet18/', dataset, file_name)
    data_d = scio.loadmat(file_path)
    te_BI = data_d['te_BI']#[:500, :]
    te_BT = data_d['te_BT']#[:500, :]
    te_L =   data_d['te_L']#[:500, :]

    db_BI = data_d['db_BI']
    db_BT = data_d['db_BT']
    db_L =   data_d['db_L']

    perplexity = 50 

    data_I, data_T, labels = filter_top_labels_and_samples(db_BI, db_BT, db_L)

    # data_I = te_BI
    # data_T = te_BT
    # labels =  te_L
    # random_indices = np.random.choice(te_BI.shape[0], size=100, replace=False)
    # data_I = te_BI[random_indices, :]
    # data_T = te_BT[random_indices, :]
    # labels =  te_L[random_indices, :]

    # sample_idx = np.sum(te_L, axis=1) == 1

    # db_BI = data_d['db_BI'][:100, :]
    # db_BT = data_d['db_BT'][:100, :]
    # db_L =   data_d['db_L'][:100, :]

    top_n_labels = 10  # 设置展示的标签数量
    # data_I = te_BI[sample_idx][:100, :]
    # data_T = te_BT[sample_idx][:100, :]
    # labels = te_L[sample_idx][:100, :]

    # data_I = np.vstack((te_BI, db_BI))
    # data_T = np.vstack((te_BT, db_BT))
    # labels = np.vstack((te_L, db_L))
    # random_indices = np.random.choice(data_I.shape[0], size=1000, replace=False)
    # data_I = data_I[random_indices]
    # data_T = data_T[random_indices]
    # labels = labels[random_indices]

    # Step 3: 统计标签的频率并选择前 N 个主要标签
    label_counts = Counter(np.argmax(labels, axis=1))  # 统计每个标签的频率
    top_labels = [label for label, _ in label_counts.most_common(top_n_labels)]  # 获取前 N 个主要标签

    # Step 4: 创建标签到连续索引的映射
    label_to_index = {lbl: i for i, lbl in enumerate(top_labels)}  # 标签 -> 索引

    # Step 5: 过滤数据，只保留包含主要标签的样本
    filtered_data_I = []  # 模态 I 的过滤数据
    filtered_data_T = []  # 模态 T 的过滤数据
    filtered_label_indices = []  # 共享的过滤标签索引

    for i, sample in enumerate(range(data_I.shape[0])):
        active_labels = np.where(labels[i] == 1)[0]  # 获取当前样本的标签
        common_labels = set(active_labels).intersection(top_labels)  # 找到与主要标签的交集
        if common_labels:
            # 模态 I 和模态 T 对应同一个样本
            filtered_data_I.append(data_I[i])  # 模态 I 数据
            filtered_data_T.append(data_T[i])  # 模态 T 数据
            filtered_label_indices.append(label_to_index[next(iter(common_labels))])  # 共享标签索引

    # print('Seleted samples:', len(filtered_data_I))

    filtered_BI = np.array(filtered_data_I)
    filtered_BT = np.array(filtered_data_T)
    filtered_label_indices = np.array(filtered_label_indices)

    # Step 6: 可视化
    # Step 1: 合并两个模态的数据
    data = np.vstack((filtered_BI, filtered_BT))  # 将两个模态拼接在一起

    # Step 2: 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate=200)
    data_2d = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))

    # 定义颜色映射
    # cmap = plt.cm.get_cmap('tab10', top_n_labels)
    # colors = [cmap(i) for i in range(top_n_labels)]

    cmap = colormaps['tab10']  # 使用新的 colormaps API
    colors = [cmap(i) for i in range(top_n_labels)]

    # 散点大小（可以根据需要调整）
    scatter_size = 100  # 调大散点符号的大小

    # 绘制模态 I 的散点图（圆形）
    scatter_I = []
    for lbl, index in label_to_index.items():
        indices = np.where(filtered_label_indices == index)[0]
        if len(indices) > 0:  # 确保有数据点
            scatter_I.append(
                plt.scatter(
                    data_2d[indices, 0], data_2d[indices, 1],
                    # color=colors[index], alpha=0.7, marker='o', s=scatter_size, label=f"Modality I - Label {lbl}"
                    color=colors[index], alpha=0.7, marker='o', s=scatter_size
                )
            )

    # 绘制模态 T 的散点图（叉号）
    scatter_T = []
    for lbl, index in label_to_index.items():
        indices = np.where(filtered_label_indices == index)[0]
        if len(indices) > 0:  # 确保有数据点
            scatter_T.append(
                plt.scatter(
                    data_2d[indices + filtered_BI.shape[0], 0], data_2d[indices + filtered_BI.shape[0], 1],
                    # color=colors[index], alpha=0.7, marker='x', s=scatter_size, label=f"Modality T - Label {lbl}"
                    color=colors[index], alpha=0.7, marker='x', s=scatter_size
                )
            )

    # plt.title(f't-SNE Visualization with Top {top_n_labels} Labels', fontsize=16)  # 调整标题字体大小
    # plt.xlabel('t-SNE Dimension 1', fontsize=14)  # 调整坐标轴标签字体大小
    # plt.ylabel('t-SNE Dimension 2', fontsize=14)  # 调整坐标轴标签字体大小

    # 添加图例
    handles = scatter_I + scatter_T
    # circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Image')
    # cross = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=10, label='Text')
    # plt.legend(handles=[circle, cross], title="Modalities", fontsize=12, title_fontsize=14, loc="upper right")

    # 添加图例
    # labels = [f"{label_names[lbl]} (Image)" for lbl in top_labels] + [f"{label_names[lbl]} (Text)" for lbl in top_labels]
    # plt.legend(handles, labels, title="Labels", fontsize=12, title_fontsize=14, loc="upper right", bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    # plt.show()

    plt.savefig(f't_sne_result/{dataset}_{file_name}.png')
    # plt.savefig(f't_sne_result/{dataset}_{file_name}.eps', format='eps', dpi=300)

    plt.close()


def main():
    # T = 4
    # bit = 128
    # dataset = 'flickr'  # flickr or coco or nuswide
    # file_name = f'{T}-{bit}-True-ours.mat'

    datasets = ['flickr', 'coco', 'nuswide']
    Ts = [1, 2, 4]
    bs = [16, 32, 64, 128]

    with tqdm(total=len(datasets) * len(Ts) * len(bs)) as pbar:
        for T in Ts:
            for dataset in datasets:
                for bit in bs:
                    file_name = f'{T}-{bit}-True-ours.mat'
                    show_t_sne(dataset, file_name)
                    pbar.update(1)

    # file_names = [
    #     '1-16-True-ours.mat',
    #     '1-32-True-ours.mat',
    #     '1-64-True-ours.mat',
    #     '1-128-True-ours.mat',
    # ]

    show_t_sne(dataset, file_name)
    # show_t_sne('nuswide', '1-128-True-ours.mat')
    # show_t_sne(dataset, '1-16-True-ours.mat')
    # for dataset in datasets:
    #     for file_name in tqdm(file_names):
    #         show_t_sne(dataset, file_name)


if __name__ == '__main__':
    main()
