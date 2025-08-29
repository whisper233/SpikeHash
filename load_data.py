import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import transforms
from PIL import Image
import h5py
import os


###############
# Imagenette #
###############
import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.imagenette import Imagenette

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=None,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation="bicubic",
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def create_imagenette_dataloader(args, is_train=True, data_dir='/data/zhangzhen/snn/ImageNetTe/'):
    transform = build_transform(is_train, args)
    dataset = Imagenette(root=data_dir, split="train" if is_train else "val", transform=transform, download=False)
    # print(dataset)
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        sampler=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )


    return data_loader_train


###############
# Flickr #
###############

g_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MyEnergyImageDataset(Dataset):
    def __init__(self, args, sample_info):
        self.args = args
        self.img_feat = sample_info['img_feat']

    def __len__(self):
        return 100

    def __getitem__(self, index):
        img = self.img_feat[index]

        return img, img


class MyEnergyTextDataset(Dataset):
    def __init__(self, args, sample_info):
        self.args = args
        self.txt = sample_info['txt'].astype(np.float32)
        

    def __len__(self):
        return 100

    def __getitem__(self, index):
        bow = self.txt[index]
        return bow, bow

class MyDataset(Dataset):
    def __init__(self, args, img_type, sample_info, img_dir, transform=None):
        self.args = args
        self.txt = sample_info['txt'].astype(np.float32)
        self.label = sample_info['label'].astype(np.float32)
        self.img_path = sample_info['img_path']
        self.img_dir = img_dir
        self.transform = transform
        self.img_type = img_type
        
        if img_type == 'img_feat':
            self.img_feat = sample_info['img_feat']

        self.length = self.label.shape[0]
        self.txt_dim = self.txt.shape[1]
        self.label_dim = self.label.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_path[index]
        bow = self.txt[index]
        label = self.label[index]
        if self.img_type == 'img_feat':
            img = self.img_feat[index]
        elif self.img_type == 'src_img':

            abs_img_path = os.path.join(self.img_dir, img_path)
            pil_img = Image.open(abs_img_path).convert("RGB") # TODO
            img = self.transform(pil_img)
            pil_img.close()
        else:
            raise NotImplementedError

        return img_path, img, bow, label
    
    def get_txt_dim(self):
        return self.txt_dim

    def get_label_dim(self):
        return self.label_dim



class MyDatasetEnergyImage(Dataset):
    def __init__(self, args, sample_info, img_dir, transform=None):
        self.args = args
        self.img_path = sample_info['img_path']
        self.img_dir = img_dir
        self.transform = transform
        
        self.length = self.img_path.shape[0]

    def __len__(self):
        # return self.length
        return 100

    def __getitem__(self, index):
        img_path = self.img_path[index]

        abs_img_path = os.path.join(self.img_dir, img_path)
        pil_img = Image.open(abs_img_path).convert("RGB") # TODO
        img = self.transform(pil_img)
        pil_img.close()

        return img, index

class MyDatasetEnergyText(Dataset):
    def __init__(self, args, sample_info):
        self.args = args
        self.txt = sample_info['txt'].astype(np.float32)
        self.length = self.txt.shape[0]

    def __len__(self):
        # return self.length
        return 100

    def __getitem__(self, index):
        bow = self.txt[index]

        return bow, index


class NuswideDataset(Dataset):
    def __init__(self, args, img, tag, label, img_dir, transform=None):
        self.args = args
        self.img = img
        self.tag = tag
        self.label = label
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.img.shape[0]    

    def __getitem__(self, index):
        img_file = self.img[index]
        tag = self.tag[index]
        label = self.label[index]

        img_path = os.path.join(self.img_dir, img_file)
        pil_img = Image.open(img_path).convert("RGB") # TODO
        img = self.transform(pil_img)

        # return index, img_feat, img, txt_feat, bow, label, sample_id
        return img, tag, label

    def get_txt_dim(self):
        return 1000

    def get_label_dim(self):
        return 21
    

class CocoDataset(Dataset):
    def __init__(self, args, img, tag, label, img_dir, transform=None):
        self.args = args
        self.img = img
        self.tag = tag
        self.label = label
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.img.shape[0]    

    def __getitem__(self, index):
        img_file = self.img[index]
        tag = self.tag[index]
        label = self.label[index]

        img_path = os.path.join(self.img_dir, img_file)
        pil_img = Image.open(img_path).convert("RGB") # TODO
        img = self.transform(pil_img)

        return img, tag, label
    
    def get_txt_dim(self):
        return 2000
    
    def get_label_dim(self):
        return 80


def create_common_dataloader(args, train_info, test_info, db_info, img_dir, cur_task):

    train_drop_last = False

    if cur_task == 'pretrain':
        img_type = 'src_img'
        train_transform = build_transform(is_train=True, args=args)
        val_transform = build_transform(is_train=False, args=args)
        train_drop_last = True

    elif cur_task == 'extract_ferat':
        img_type = 'src_img'
        train_transform = build_transform(is_train=False, args=args)
        val_transform = build_transform(is_train=False, args=args)
        train_drop_last = False

    elif cur_task == 'hash':
        
        if args.freeze_img_backbone: # 如果是锁定backbone的话，那么读取原图时，就不需要再用train的transform
            img_type = 'img_feat'
            train_transform = None
            val_transform = None
            train_drop_last = True

        else:   # 微调backbone
            img_type = 'src_img'
            train_transform = build_transform(is_train=True, args=args)
            val_transform = build_transform(is_train=False, args=args)
            train_drop_last = True

    else:
        raise NotImplementedError

    tr_dataset = MyDataset(args, img_type, train_info, img_dir, transform=train_transform)
    te_dataset = MyDataset(args, img_type, test_info, img_dir, transform=val_transform)
    db_dataset = MyDataset(args, img_type, db_info, img_dir, transform=val_transform)

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=train_drop_last)
    te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    db_loader = DataLoader(db_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    return tr_loader, te_loader, db_loader



def create_energy_dataloader(args, train_info, img_dir):

    val_transform = build_transform(is_train=False, args=args)
    img_dataset = MyDatasetEnergyImage(args, train_info, img_dir, transform=val_transform)
    txt_dataset = MyDatasetEnergyText(args, train_info)

    img_loader = DataLoader(img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    txt_loader = DataLoader(txt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return img_loader, txt_loader


def load_sample_info(args):
    if args.dataset == "flickr":
        # img_dir = "/data/zz/CMR/data/raw/flickr/mirflickr"
        img_dir = "/home/zhangzhen/tmp_data/flickr/mirflickr"
        train_info_path = "/data/zhangzhen/CMR/data/feat/flickr/flickr_train_sample_info.npy" # ['img_path', 'label', 'txt']
        test_info_path = "/data/zhangzhen/CMR/data/feat/flickr/flickr_test_sample_info.npy" # ['img_path', 'label', 'txt']
        db_info_path = "/data/zhangzhen/CMR/data/feat/flickr/flickr_train_sample_info.npy"

    elif args.dataset == "nuswide":
        img_dir = "/data/zhangzhen/CMR/data/raw/nuswide/Flickr"
        # img_dir = "/home/zhangzhen/tmp_data/nuswide/Flickr"
        train_info_path = "/data/zhangzhen/CMR/data/feat/nuswide/nuswide_train_sample_info.npy" # ['img_path', 'label', 'txt']
        test_info_path = "/data/zhangzhen/CMR/data/feat/nuswide/nuswide_test_sample_info.npy" # ['img_path', 'label', 'txt']
        db_info_path = "/data/zhangzhen/CMR/data/feat/nuswide/nuswide_train_sample_info.npy"

    elif args.dataset == "coco":
        img_dir = "/data/zhangzhen/CMR/data/raw/coco"
        # img_dir = "/home/zhangzhen/tmp_data/coco"
        train_info_path = "/data/zhangzhen/CMR/data/feat/coco/coco_train_sample_info.npy" # ['img_path', 'label', 'txt']
        test_info_path = "/data/zhangzhen/CMR/data/feat/coco/coco_test_sample_info.npy" # ['img_path', 'label', 'txt']
        db_info_path = "/data/zhangzhen/CMR/data/feat/coco/coco_train_sample_info.npy"

    else:
        raise NotImplementedError
    
    train_info = np.load(train_info_path, allow_pickle=True).item()
    test_info = np.load(test_info_path, allow_pickle=True).item()
    db_info = np.load(db_info_path, allow_pickle=True).item()

    return train_info, test_info, db_info, img_dir

# cur_task： 'pretrain', 'extract_ferat', 'hash'
# def create_common_dataloader(args, train_info, test_info, db_info, img_dir='', cur_task='pretrain'):
#     if args.dataset == "flickr":
#         return create_common_dataloader_haha(args, train_info, test_info, db_info, img_dir, cur_task)
#     elif args.dataset == "nuswide":
#         # return create_nuswide_dataloader(args)
#         return create_common_dataloader_haha(args, train_info, test_info, db_info, img_dir, cur_task)
#     elif args.dataset == "coco":
#         # return create_coco_dataloader(args)
#         return create_common_dataloader_haha(args, train_info, test_info, db_info, img_dir, cur_task)
#     else:
#         raise NotImplementedError

