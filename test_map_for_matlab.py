import torch
import numpy as np
from scipy.io import savemat, loadmat
from hash_val import calculate_top_map


def create_data():
    values = [-1, 1]
    te = np.random.choice(values, size=(1000, 32)).astype(np.float32)
    db = np.random.choice(values, size=(10000, 32)).astype(np.float32)

    values = [0, 1]
    p = [0.9, 0.1]
    te_l = np.random.choice(values, size=(1000, 24), p=p).astype(np.float32)
    db_l = np.random.choice(values, size=(10000, 24), p=p).astype(np.float32)

    # 创建一些示例数据
    data = {
        'te':te,
        'db':db,
        'te_l':te_l,
        'db_l':db_l
    }

    return data
    # 保存为 .mat 文件
    # savemat('output.mat', data)


def main():
    data_path = 'output.mat'
    
    # data = create_data()
    # savemat(data_path, data)

    data = loadmat(data_path)

    te = torch.from_numpy(data['te']).cuda()
    db = torch.from_numpy(data['db']).cuda()

    te_l = torch.from_numpy(data['te_l']).cuda()
    db_l = torch.from_numpy(data['db_l']).cuda()


    matrix = calculate_top_map(te, db, te_l, db_l, topk=50)

    print(matrix)

    pass

if __name__ == '__main__':
    main()
