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
from PIL import Image

from matplotlib import font_manager

# 添加自定义字体路径
font_path = "/home/zhangzhen/.fonts/times.ttf"  # 替换为实际字体文件路径
font_manager.fontManager.addfont(font_path)

plt.rc('font', family='Times New Roman')

def filter_samples(src_BI, src_BT, src_L, N=100):
    """
    筛选单标签数据，选择10个最常用的标签，并为每个标签选择100个样本。
    
    参数:
        te_BI (numpy.ndarray): 图像特征矩阵，形状为 (n_samples, n_image_features)
        te_BT (numpy.ndarray): 文本特征矩阵，形状为 (n_samples, n_text_features)
        te_L (numpy.ndarray): 标签矩阵，形状为 (n_samples, n_classes)，one-hot编码
    
    返回:
        filtered_BI (numpy.ndarray): 过滤后的图像特征矩阵
        filtered_BT (numpy.ndarray): 过滤后的文本特征矩阵
        filtered_L (numpy.ndarray): 过滤后的标签矩阵
    """
    # 获取样本数和类别数
    n_samples, n_classes = src_L.shape

    # 1. 筛选出所有单标签样本
    single_label_indices = []
    for i in range(n_samples):
        if np.sum(src_L[i]) == 1:  # 如果样本只有一个标签
            single_label_indices.append(i)

    single_label_BI = src_BI[single_label_indices]
    single_label_BT = src_BT[single_label_indices]
    single_label_L = src_L[single_label_indices]

    # 2. 统计每个标签的使用频率
    label_counts = np.sum(single_label_L, axis=0)
    sorted_labels = np.argsort(label_counts)[::-1]  # 按标签使用频率从高到低排序

    # 3. 选择前10个标签
    top_10_labels = sorted_labels[:10]

    # 4. 为每个标签选择N个样本
    filtered_indices = []
    for label in top_10_labels:
        label_indices = np.where(single_label_L[:, label] == 1)[0]  # 当前标签的样本索引
        selected_indices = label_indices[:N]  # 取前N个样本
        filtered_indices.extend(selected_indices)

    # 提取过滤后的样本
    filtered_BI = single_label_BI[filtered_indices]
    filtered_BT = single_label_BT[filtered_indices]
    filtered_L = single_label_L[filtered_indices]

    return filtered_BI, filtered_BT, filtered_L

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_tsne(filtered_BI, filtered_BT, filtered_L, perplexity, save_path):
    """
    使用t-SNE对过滤后的特征进行可视化。图像和文本特征用不同符号表示，但相同样本共享相同颜色。
    
    参数:
        filtered_BI (numpy.ndarray): 过滤后的图像特征矩阵
        filtered_BT (numpy.ndarray): 过滤后的文本特征矩阵
        filtered_L (numpy.ndarray): 过滤后的标签矩阵
    """
    # 获取标签，并将其转换为类别的索引
    labels = np.argmax(filtered_L, axis=1)

    # 合并图像和文本特征
    combined_features = np.concatenate([filtered_BI, filtered_BT], axis=0)

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded_features = tsne.fit_transform(combined_features)

    # 分离图像和文本的嵌入
    embedded_BI = embedded_features[:len(filtered_BI)]  # 前半部分是图像特征
    embedded_BT = embedded_features[len(filtered_BI):]  # 后半部分是文本特征

    # 自定义颜色映射
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # 使用tab10色系


    # 绘制图像特征和文本特征
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        color = colors[i]  # 当前类别的颜色
        
        # 绘制图像特征
        plt.scatter(embedded_BI[labels == label, 0], embedded_BI[labels == label, 1], s=200, facecolors='none',
                    color=color, alpha=0.7, marker='o')
        
        # 绘制文本特征
        plt.scatter(embedded_BT[labels == label, 0], embedded_BT[labels == label, 1], s=200,
                    color=color, alpha=0.7, marker='x')


    plt.scatter([], [], color='black', alpha=0.7, marker='o', label="Image", s=200, facecolors='none')
    plt.scatter([], [], color='black', alpha=0.7, marker='x', label="Text", s=200)

    # 调整图例以避免重复
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=20, loc='lower right')
    
    # 设置 x 轴和 y 轴刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title("t-SNE Visualization of Image and Text ")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.show()
    plt.savefig(save_path+'.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(save_path)
    plt.close()


def visualize_tsne_src(filtered_BI, filtered_BT, filtered_L, perplexity, save_path):
    """
    使用t-SNE对过滤后的特征进行可视化。图像和文本特征用不同符号表示，但相同样本共享相同颜色。
    
    参数:
        filtered_BI (numpy.ndarray): 过滤后的图像特征矩阵
        filtered_BT (numpy.ndarray): 过滤后的文本特征矩阵
        filtered_L (numpy.ndarray): 过滤后的标签矩阵
    """
    # 获取标签，并将其转换为类别的索引
    labels = np.argmax(filtered_L, axis=1)

    # 合并图像和文本特征
    # combined_features = np.concatenate([filtered_BI, filtered_BT], axis=0)

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    print('processing Image...', end='')
    embedded_BI = tsne.fit_transform(filtered_BI)
    print('done')
    print('processing Text...', end='')
    embedded_BT = tsne.fit_transform(filtered_BT)
    print('done')

    # 自定义颜色映射
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # 使用tab10色系


    # 绘制图像特征和文本特征
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        color = colors[i]  # 当前类别的颜色
        
        # 绘制图像特征
        plt.scatter(embedded_BI[labels == label, 0], embedded_BI[labels == label, 1], s=200, facecolors='none',
                    color=color, alpha=0.7, marker='o')
        
        # 绘制文本特征
        plt.scatter(embedded_BT[labels == label, 0], embedded_BT[labels == label, 1], s=200,
                    color=color, alpha=0.7, marker='x')


    plt.scatter([], [], color='black', alpha=0.7, marker='o', label="Image", s=200, facecolors='none')
    plt.scatter([], [], color='black', alpha=0.7, marker='x', label="Text", s=200)

    # 调整图例以避免重复
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=20, loc='lower right')
    
    # 设置 x 轴和 y 轴刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title("t-SNE Visualization of Image and Text ")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.show()
    plt.savefig(save_path+'.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(save_path)
    plt.close()


def src_load_sample_info():
    train_info = np.load('/data/zhangzhen/CMR/data/feat/coco/coco_train_sample_info.npy', allow_pickle=True).item()
    test_info = np.load('/data/zhangzhen/CMR/data/feat/coco/coco_test_sample_info.npy', allow_pickle=True).item()

    train_path = train_info['img_path']
    train_label = train_info['label']
    train_txt = train_info['txt']

    test_path = test_info['img_path']
    test_label = test_info['label']
    test_txt = test_info['txt']

    tot_path = np.concatenate((train_path, test_path), axis=0)
    tot_label = np.concatenate((train_label, test_label), axis=0)
    tot_txt = np.concatenate((train_txt, test_txt), axis=0)

    return tot_path, tot_txt, tot_label, '/data/zhangzhen/CMR/data/raw/coco/'

def open_images(paths, img_dir):
    res = list()
    save_shape = 0

    for i in tqdm(range(len(paths))):
        img = Image.open(os.path.join(img_dir, paths[i])).convert("RGB")
        img = img.resize((224, 224))
        res.append(np.array(img))

    res = np.stack(res)
    res = res.reshape(res.shape[0], -1)

    return res

def show_t_sne_src():
    # file_path = os.path.join('/data/zhangzhen/logs/back/torch_resnet/', dataset, method, file_name)
    # file_path = os.path.join('/data/zhangzhen/logs/back/torch_resnet/', dataset, 'DCHMT', file_name)
    # file_path = os.path.join('/data/zhangzhen/logs/resnet18/', dataset, file_name)

    tot_path, tot_txt, tot_label, img_dir = src_load_sample_info()

    indices = np.arange(len(tot_label))  # 获取样本索引 [0, 1, 2, ..., n-1]
    np.random.shuffle(indices)       # 随机打乱索引
    filtered_path, T, L = filter_samples(tot_path, tot_txt, tot_label, 50)

    I = open_images(filtered_path, img_dir)



    # for perplexity in range(5, 51, 5):
    visualize_tsne_src(I, T, L, perplexity=50, save_path=f't_sne_result_all_set_50/src.png')
    # visualize_tsne(filtered_BI, filtered_BT, filtered_L, perplexity=50, save_path=f't_sne_other_method_result_all_set_50/coco/tsne_{method}.png')
        # visualize_tsne(filtered_BI, filtered_BT, filtered_L, perplexity, f't_sne_result_test_set_50/{dataset}-{file_name}-p-{perplexity}.png')
        # visualize_tsne(filtered_BI, filtered_BT, filtered_L, perplexity, f't_sne_result_all_set_50/{dataset}-{file_name}-p-{perplexity}.png')

def show_t_sne(dataset, file_name):
    # file_path = os.path.join('/data/zhangzhen/logs/back/torch_resnet/', dataset, method, file_name)
    # file_path = os.path.join('/data/zhangzhen/logs/back/torch_resnet/', dataset, 'DCHMT', file_name)
    file_path = os.path.join('/data/zhangzhen/logs/resnet18/', dataset, file_name)
    data_d = scio.loadmat(file_path)
    te_BI = data_d['te_BI'][:5000, :]
    te_BT = data_d['te_BT'][:5000, :]
    te_L =   data_d['te_L'][:5000, :]

    db_BI = data_d['db_BI']
    db_BT = data_d['db_BT']
    db_L =   data_d['db_L']

    # BI = te_BI
    # BT = te_BT
    # L = te_L

    BI = np.vstack((te_BI, db_BI))
    BT = np.vstack((te_BT, db_BT))
    L = np.vstack((te_L, db_L))

    indices = np.arange(len(L))  # 获取样本索引 [0, 1, 2, ..., n-1]
    np.random.shuffle(indices)       # 随机打乱索引

    BI = BI[indices]
    BT = BT[indices]
    L = L[indices]

    filtered_BI, filtered_BT, filtered_L = filter_samples(BI, BT, L, 50)

    # for perplexity in range(5, 51, 5):
    visualize_tsne(filtered_BI, filtered_BT, filtered_L, perplexity=50, save_path=f't_sne_result_all_set_50/{file_name}-p-{50}.png')
    # visualize_tsne(filtered_BI, filtered_BT, filtered_L, perplexity=50, save_path=f't_sne_other_method_result_all_set_50/coco/tsne_{method}.png')
        # visualize_tsne(filtered_BI, filtered_BT, filtered_L, perplexity, f't_sne_result_test_set_50/{dataset}-{file_name}-p-{perplexity}.png')
        # visualize_tsne(filtered_BI, filtered_BT, filtered_L, perplexity, f't_sne_result_all_set_50/{dataset}-{file_name}-p-{perplexity}.png')

def main():
    # datasets = ['coco', 'flickr', 'nuswide']
    # Ts = [1, 2, 4]
    # bs = [16, 32, 64, 128]

    # datasets = ['nuswide']
    # Ts = [16, 32, 64, 128]
    # bs = [16]

    # with tqdm(total=len(datasets) * len(Ts) * len(bs)) as pbar:
    #     for dataset in datasets:
    #         for T in Ts:
    #             for bit in bs:
    #                 file_name = f'{T}-{bit}-True-ours.mat'
    #                 show_t_sne(dataset, file_name)
    #                 pbar.update(1)


    # datasets = ['coco', 'flickr', 'nuswide']
    # datasets = ['coco']
    # methods = ['DCHMT' , 'DCHUC' , 'DNpH' , 'DSPH' , 'HMAH' , 'MIAN' , 'SCRATCH']
    # bs = [128]

    # with tqdm(total=len(datasets) * len(methods) * len(bs)) as pbar:
    #     for dataset in datasets:
    #         for method in methods:
    #             for bit in bs:
    #                 file_name = f'{bit}.mat'
    #                 show_t_sne(dataset, method, file_name)
    #                 pbar.update(1)
    while True:
        # show_t_sne('coco', '1-128-True-ours.mat')
        show_t_sne_src()


if __name__ == "__main__":
    main()
