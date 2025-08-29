import os
import re
import shutil
import time
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from PIL import Image
import matplotlib.pyplot as plt

from spikingjelly.clock_driven import functional as snn_functional

import json
from hash_model import build_model
from hash_val import calculate_hamming


# label = ['sea', 'water', 'plant life', 'indoor', 'tree', 'lake', 'river', 'male', 'bird', 'portrait', 'transport', 'baby', 
#          'night', 'structures', 'female', 'flower', 'people', 'clouds', 'animals', 'food', 'sunset', 'car', 'dog', 'sky']

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    cuda_id = "3"
    print("auto set cuda id %s" % cuda_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

class TmpArgs():
    def __init__(self):
        self.img_backbone = "resnet18"
        self.img_T = 1
        self.txt_T = 1
        self.fusion_mode = 'ours'
        self.bit_len = 16
        self.txt_dim = 2000
        self.label_dim = 80
        self.dataset = 'coco'

        self.query_num = 5000
        self.topN = 5

        self.img_dir = '/data/zhangzhen/CMR/data/raw/coco/'
        # self.txt_dir = '/home/zhangzhen/tmp_data/flickr/mirflickr/meta/tags'
        # self.img_dir = '/home/zhangzhen/tmp_data/flickr/mirflickr'
        # self.txt_dir = '/home/zhangzhen/tmp_data/flickr/mirflickr/meta/tags'

class TmpDataset(Dataset):
    def __init__(self, sample_info):
        self.img_path = sample_info['img_path']
        self.img_feat = sample_info['img_feat']
        self.txt = sample_info['txt'].astype(np.float32)
        self.label = sample_info['label'].astype(np.float32)
        self.length = self.label.shape[0]
        self.txt_dim = self.txt.shape[1]
        self.label_dim = self.label.shape[1]

    def __len__(self):
        return self.txt.shape[0]

    def __getitem__(self, index):
        return self.img_path[index], self.img_feat[index], self.txt[index], self.label[index]


def load_state_dict(img_model, txt_model, dataset):
    ckp_path = f'/data/zhangzhen/logs/resnet18/{dataset}/1-16-True-ours.pth'

    ckp = torch.load(ckp_path)
    img_model.load_state_dict(ckp['img_model'])
    txt_model.load_state_dict(ckp['txt_model'])

def create_tmp_dataloader(dataset, T):
    test_feat_path =  f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_spiking_resnet_spike_feat/test_feat_spiking_resnet18_T{T}.npy'
    db_feat_path =  f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_spiking_resnet_spike_feat/train_feat_spiking_resnet18_T{T}.npy'

    test_sample_info = np.load(test_feat_path, allow_pickle=True).item()
    db_sample_info = np.load(db_feat_path, allow_pickle=True).item()

    test_dataset = TmpDataset(test_sample_info)
    db_dataset = TmpDataset(db_sample_info)

    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    db_dataloader = DataLoader(db_dataset, batch_size=100, shuffle=False, num_workers=0)

    return test_dataloader, db_dataloader

# def read_txt(file_path):

#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = [line.strip() for line in file.readlines()]
    
#     concatenated_string = ' '.join(lines)

#     return concatenated_string


# def flickr_find_txt_by_img_path(img_path, txt_dir):
#     numbers = re.findall(r'\d+', img_path)
#     sample_id = int(numbers[0])
#     txt_file_path = f'tags{sample_id}.txt'
#     txt_file_abs_path = os.path.join(txt_dir, txt_file_path)

#     txt = read_txt(txt_file_abs_path)

#     return txt_file_path, txt_file_abs_path, txt


def coco_find_txt_by_img_path(caption_dict, img_path):
    return caption_dict[img_path]


def get_sample_captions(caption_json_paths):
    all_captions = dict() # {id, caption}
    for json_path in caption_json_paths:
        if 'train' in json_path:
            format = 'train2017/%012d.jpg'
        elif 'val' in json_path:
            format = 'val2017/%012d.jpg'

        with open(json_path, 'r') as f:                                                                                                                                                                            
            json_data = json.load(f)
            annotations = json_data['annotations']
                
            for item in tqdm(annotations):
                image_id = item['image_id']
                image_path = format % image_id

                caption = item['caption'].strip()
                if image_path not in all_captions.keys():
                    all_captions[image_path] = ""
                if not caption.endswith('.'):
                    caption = caption + '.'
                all_captions[image_path] = all_captions[image_path] + " " + caption

    return all_captions

def get_sample_label(instances_json_paths):                                                                                                                                                                        
    
    all_label = OrderedDict() # {id, label_id}
    for json_path in instances_json_paths:
        if 'train' in json_path:
            format = 'train2017/%012d.jpg'
        elif 'val' in json_path:
            format = 'val2017/%012d.jpg'

        with open(json_path, 'r') as f:
            json_data = json.load(f)
            annotations = json_data['annotations']
            for item in tqdm(annotations):
                image_id = item['image_id']
                image_path = format % image_id
                label_id = item['category_id']

                if image_path not in all_label.keys():
                    all_label[image_path] = []

                all_label[image_path].append(label_id)

    return all_label

def inference(img_model, txt_model, loader):
    all_path_l = list()
    B_I_l = list()
    B_T_l = list()
    label_l = list()

    for i, (img_path, img_feat, txt, label) in tqdm(enumerate(loader)):

        img_feat = img_feat.permute(1, 0, 2)
        img_feat = img_feat.cuda()
        txt = txt.cuda()

        H_I = img_model(img_feat).mean(0).detach()
        H_T = txt_model(txt).mean(0).detach()

        B_I = H_I.sign().cpu()
        B_T = H_T.sign().cpu()

        all_path_l.extend(img_path)
        B_I_l.append(B_I)
        B_T_l.append(B_T)
        label_l.append(label)

        # print(i, img_path, H_I, H_T, B_I, B_T, label)

        snn_functional.reset_net(img_model)
        snn_functional.reset_net(txt_model)
        torch.cuda.empty_cache()

    all_B_I = torch.cat(B_I_l)
    all_B_T = torch.cat(B_T_l)
    all_label = torch.cat(label_l)

    return all_path_l, all_B_I, all_B_T, all_label
    # return all_path_l,None, None, None


def get_label_name(label, one_hot):
    idxs = torch.where(one_hot==1)[0].tolist()
    names_l = [label[i] for i in idxs]

    names = ' '.join(names_l)

    return names


def judge_sim(test_labels, db_labels,S, i, j):
    test_label = test_labels[i]
    db_label = db_labels[j]

    query_label_num = len(torch.where(test_label==1)[0].tolist())
    db_label_num = len(torch.where(db_label==1)[0].tolist())

    # if query_label_num > 1:
    share_label_num = S[i, j]

    if abs(share_label_num - query_label_num) < 2 and share_label_num >= 2: 
        return True
    # else:
    #     if S[i, j] != 1 and db_label_num < 3:
    #         return True
    
    # if S[i, j] == 1 and query_label_num > 1 and db_label_num > 1:

    return False

def judge_sim_I2T(test_label, db_label,S, i, j):
    return judge_sim(test_label, db_label,S, i, j)


def judge_sim_T2I(test_label, db_label,S, i, j):
    return judge_sim(test_label, db_label,S, i, j)

if __name__ == "__main__":
    args = TmpArgs()
    I2T_cnt = 0
    T2I_cnt = 0

    img_model, txt_model, fusion_model = build_model(args, txt_dim=args.txt_dim, label_dim=args.label_dim, img_model_type='head')

    load_state_dict(img_model, txt_model, args.dataset)

    img_model = img_model.cuda()
    txt_model = txt_model.cuda()

    img_model.eval()
    txt_model.eval()
    
    test_dataloader, db_dataloader = create_tmp_dataloader(args.dataset, args.img_T)

    test_path, test_B_I, test_B_T, test_label = inference(img_model, txt_model, test_dataloader)
    db_path, db_B_I, db_B_T, db_label = inference(img_model, txt_model, db_dataloader)

    hamming_dist_I2T = calculate_hamming(test_B_I, db_B_T)
    hamming_dist_T2I = calculate_hamming(test_B_T, db_B_I)

    I2T_top10_id = torch.argsort(hamming_dist_I2T, dim=1)[:, :10]
    T2I_top10_id = torch.argsort(hamming_dist_T2I, dim=1)[:, :10]

    print('calc S... ', end='')
    S = test_label @ db_label.T
    # S[S>0] = 1
    print('Done')

    json_dir = "/data/zhangzhen/CMR/data/raw/coco/annotations/my_json"
    train_caption_json_path = os.path.join(json_dir, "my_captions_train2017.json")                                                                                                                                 
    val_caption_json_path = os.path.join(json_dir, "my_captions_val2017.json")
    all_captions = get_sample_captions([train_caption_json_path, val_caption_json_path])

    # json_dir = "/data/zhangzhen/CMR/data/raw/coco/annotations/my_json"
    # train_instances_json_path = os.path.join(json_dir, "my_instances_train2017.json")
    # val_instances_json_path = os.path.join(json_dir, "my_instances_val2017.json")                                                                                                                                  
    # all_label = get_sample_label([train_instances_json_path, val_instances_json_path])

    
    # I2T
    for i in tqdm(range(args.query_num)):

        cur_sample_path = test_path[i]

        labels_index = torch.where(test_label[i]==1)[0].tolist()

        cnt = 0
        righr_cnt = 0
        for j in range(args.topN):
            cur_I2T_id = I2T_top10_id[i][j]
            if judge_sim_I2T(test_label, db_label, S, i, cur_I2T_id):
                cnt += 1
            if S[i, cur_I2T_id] != 0:
                righr_cnt += 1
        if cnt < args.topN - 2 or righr_cnt != args.topN:
            continue

        I2T_cnt += 1
        cur_label_names = '-'.join([COCO_CLASSES[l_i] for l_i in labels_index])

        file_name = cur_sample_path.split('/')[-1]

        abs_sample_path = os.path.join(args.img_dir, cur_sample_path)

        cur_dir = f'./tmp/I2T/{i}-{cur_label_names}/'
        os.makedirs(cur_dir, exist_ok=False)

        shutil.copy(abs_sample_path, f'{cur_dir}/{file_name}')
        

        for j in range(args.topN):
            cur_I2T_id = I2T_top10_id[i][j]

            correct = S[i][cur_I2T_id] != 0

            cur_db_path = db_path[cur_I2T_id]
            cur_db_txt = coco_find_txt_by_img_path(all_captions, cur_db_path)
            
            txt_file_path = os.path.join(cur_dir, f'db.txt')
            with open(txt_file_path, 'a', encoding='utf-8') as file:
                file.write(cur_db_txt + '\n')

    # T2I
    for i in tqdm(range(args.query_num)):

        cur_sample_path = test_path[i]

        labels_index = torch.where(test_label[i]==1)[0].tolist()

        cnt = 0
        righr_cnt = 0
        for j in range(args.topN):
            cur_T2I_id = T2I_top10_id[i][j]
            if judge_sim_T2I(test_label, db_label, S, i, cur_T2I_id):
                cnt += 1
            if S[i][cur_T2I_id] != 0:
                righr_cnt += 1
        if cnt < args.topN - 1 or righr_cnt != args.topN:
            continue
        
        T2I_cnt += 1

        cur_label_names = '-'.join([COCO_CLASSES[l_i] for l_i in labels_index])
        file_name = cur_sample_path.split('/')[-1]

        abs_sample_path = os.path.join(args.img_dir, cur_sample_path)

        cur_dir = f'./tmp/T2I/{i}-{cur_label_names}/'
        os.makedirs(cur_dir, exist_ok=False)
        cur_test_txt = coco_find_txt_by_img_path(all_captions, cur_sample_path)

        txt_file_path = os.path.join(cur_dir, f'{j}.txt')
        with open(txt_file_path, 'w', encoding='utf-8') as file:
            file.write(cur_test_txt)
        
        for j in range(args.topN):
            cur_T2I_id = T2I_top10_id[i][j]
            
            correct = S[i][cur_T2I_id] != 0

            cur_db_path = db_path[cur_T2I_id]
            img_abs_path = os.path.join(args.img_dir, cur_db_path)
            
            file_name = cur_db_path.split('/')[-1]
            shutil.copy(img_abs_path, f'{cur_dir}/db_{j}-{correct}-{file_name}')


    print('I2T_cnt: ', I2T_cnt)
    print('T2I_cnt: ', T2I_cnt)
