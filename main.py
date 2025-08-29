import os
import re
import time
import copy
import random
import argparse
# from snoop import snoop
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import functional
from scipy.io import savemat

import wandb
from tqdm import tqdm
from syops import get_model_complexity_info

from hash_val import validate
from hash_model import build_model
from load_data import create_common_dataloader, load_sample_info, create_energy_dataloader, MyEnergyImageDataset, MyEnergyTextDataset
from print_tools import my_print
from torch.nn import CosineEmbeddingLoss
from hash_model import build_img_backbone, lood_pretrain_ckp, ImageHashHead, TxtModel



if "CUDA_VISIBLE_DEVICES" not in os.environ:
    cuda_id = "1"
    my_print("auto set cuda id %s" % cuda_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
 

os.environ["WANDB_MODE"] = "disabled"

torch.multiprocessing.set_sharing_strategy('file_system')
# torch.backends.cudnn.benchmark = True

Loss_l2 = torch.nn.MSELoss()
Loss_l1 = torch.nn.L1Loss()
criterion_CE = nn.CrossEntropyLoss()
criterion_cos = CosineEmbeddingLoss()
Sigmoid = nn.Sigmoid()
# torch.autograd.set_detect_anomaly(True)


def setup_seed(seed):
    # Python 随机数生成器
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # NumPy 随机数生成器
    np.random.seed(seed)
    
    # PyTorch 随机数生成器（CPU）
    torch.manual_seed(seed)
    
    # PyTorch 随机数生成器（GPU）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 关闭cuDNN的自动调优功能，主要是为了锁随机数


def parse_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument("--seed", type=int, default=42, metavar="int", help="random seed (default: 42)")
    parser.add_argument("--img_T", type=int, default=1, metavar="int", help="simulation time step of spiking neuron")
    parser.add_argument("--txt_T", type=int, default=1, metavar="int", help="simulation time step of spiking neuron")
    parser.add_argument("--dataset", type=str, default="flickr", metavar="str", help="flickr/nuswide/coco")
    parser.add_argument("--cls_ckp_dir", type=str, default='/data/zhangzhen/ckp/2.25/flickr/T1/resnet18', metavar="path", help="The directory  of class model ckp")
    parser.add_argument("--cls_ckp_specific_epoch", type=int, default=None, help="If not None, load the checkpoing of specific epoch in cls_ckp_dir")
    parser.add_argument("--epoch", type=int, default=100, metavar="int", help="epoch")
    parser.add_argument("--input_size", type=int, default=224, metavar="int", help="input_size")
    parser.add_argument("--task", type=str, default="cal_energy", metavar="str", help="pretrain/hash/full/cal_energy")
    # parser.add_argument("--direct_train_hash", action='store_true', help="direct train hash with out pre-train")
    parser.add_argument("--freeze_img_backbone", action='store_true', help="freeze the image model backbone?")
    
    # parser.add_argument("--lif_type", type=str, default='TET', help="TET/spikingjelly")

    parser.add_argument("--img_backbone", type=str, default='resnet18', metavar="str", help="resnet18/sew_resnet18/resnet50/sew_resnet50/meta_former")
    parser.add_argument("--bit_len", type=int, default=16, metavar="int", help="The length of hash bit")
    parser.add_argument("--fusion", action='store_true', help="Use Fusion?")

    # optimizer
    parser.add_argument("--pretrain_scheduler_lr", action='store_true', help="") # 使用后降性能
    parser.add_argument("--pretrain_lr", type=float, default=0.05, metavar="Float", help="backbone pretrain lr when pre-training")

    parser.add_argument("--img_backbone_lr", type=float, default=0.0001, metavar="Float", help="backbone fine tuning lr") # 没效果
    parser.add_argument("--hash_scheduler_lr", action='store_false', help="") # 意义不大
    parser.add_argument("--image_head_lr", type=float, default=0.05, metavar="Float", help="hash layer lr")
    parser.add_argument("--txt_lr", type=float, default=0.05, metavar="Float", help="hash func lr")
    parser.add_argument("--fusion_lr", type=float, default=0.05, metavar="Float", help="fusion model lr")
    parser.add_argument("--weight_decay", type=float, default=0, metavar="Float", help="weight_decay")
    parser.add_argument("--loss_modal_sim_factor", type=float, default=1, metavar="Float", help="")
    parser.add_argument("--loss_fusion_sim_factor", type=float, default=1, metavar="Float", help="")
    parser.add_argument("--loss_align_factor", type=float, default=0.1, metavar="Float", help="")
    
    # train & eval
    parser.add_argument("--batch_size", type=int, default=32, metavar="int", help="Batch Size")
    parser.add_argument("--num_workers", type=int, default=8, metavar="int", help="Number of dataloader workers")
    parser.add_argument("--eval_interval", type=int, default=2, metavar="int", help="Eval interval")
    parser.add_argument("--topk", type=int, default=500, metavar="int", help="The length of hash bit (default: 500)")
    # TODO
    # parser.add_argument("--not_img_pretrain", action='store_false', help="defalt")
    parser.add_argument("--tqdm_disable", action='store_true', help="Don't print the process bar")

    args = parser.parse_args()

    return args


def cal_hash_loss(args, img_out, txt_out, fusion_out, label):

    eps = torch.tensor(1e-5).cuda(non_blocking=True)

    # 相似度约束
    img_out = torch.where(img_out == 0, eps, img_out)
    txt_out = torch.where(txt_out == 0, eps, txt_out)
    fusion_out = torch.where(fusion_out == 0, eps, fusion_out)
    
    label_sim_matrix = torch.mm(label, label.T).cuda()
    label_sim_matrix[label_sim_matrix > 0] = 1

    img_normalized_matrix_code = img_out / img_out.norm(dim=1, keepdim=True)
    txt_normalized_matrix_code = txt_out / txt_out.norm(dim=1, keepdim=True)
    fusion_normalized_matrix_code = fusion_out / fusion_out.norm(dim=1, keepdim=True)

    img_sim_matrix = torch.mm(img_normalized_matrix_code, img_normalized_matrix_code.t())
    txt_sim_matrix = torch.mm(txt_normalized_matrix_code, txt_normalized_matrix_code.t())
    fusion_sim_matrix = torch.mm(fusion_normalized_matrix_code, fusion_normalized_matrix_code.t())
    
    loss_sim_img = Loss_l2(img_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())
    loss_sim_txt = Loss_l2(txt_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())
    loss_sim_fusion = Loss_l2(fusion_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())

    target = torch.ones(img_out.size(0)).cuda()  # 目标是1，表示希望两个向量尽可能相似
    if args.fusion:
        # loss_align = criterion_cos(img_out, fusion_out, target) + criterion_cos(txt_out, fusion_out, target)
        loss_align = criterion_cos(img_out, fusion_out.detach(), target) + criterion_cos(txt_out, fusion_out.detach(), target)
        # loss_align = Loss_l2(img_out, fusion_out.detach()) + Loss_l2(txt_out, fusion_out.detach())
    else:
        loss_align = criterion_cos(img_out, txt_out, target)
        # loss_align = Loss_l2(img_out, txt_out)

    # loss_reserve = criterion_cos(fusion_img, img_out.detach(), target) + criterion_cos(fusion_txt, txt_out.detach(), target)

    # 量化误差
    loss_qua = Loss_l2(img_out - img_out.sign(), torch.zeros_like(img_out).cuda()) + \
               Loss_l2(txt_out - txt_out.sign(), torch.zeros_like(txt_out).cuda()) + \
               Loss_l2(fusion_out - fusion_out.sign(), torch.zeros_like(fusion_out).cuda())
    
    if args.fusion:
        loss = loss_sim_fusion * args.loss_fusion_sim_factor + \
                loss_sim_img * args.loss_modal_sim_factor + \
                loss_sim_txt * args.loss_modal_sim_factor + \
                loss_align * args.loss_align_factor  + loss_qua * 0
    else:
        loss = loss_sim_fusion * 0 + \
                loss_sim_img * args.loss_modal_sim_factor + \
                loss_sim_txt * args.loss_modal_sim_factor + \
                loss_align * args.loss_align_factor  + loss_qua * 0

    loss_log = {'loss_sim_fusion': loss_sim_fusion.item(), 'loss_sim_img': loss_sim_img.item(), 'loss_sim_txt': loss_sim_txt.item(), \
                'loss_align': loss_align.item(), 'loss_qua': loss_qua.item(), 'loss_total': loss.item()}

    return loss, loss_log


# 稳定版
# def cal_hash_loss(args, img_out, txt_out, fusion_out, label):

#     eps = torch.tensor(1e-5).cuda(non_blocking=True)

#     # 相似度约束
#     img_out = torch.where(img_out == 0, eps, img_out)
#     txt_out = torch.where(txt_out == 0, eps, txt_out)
#     fusion_out = torch.where(fusion_out == 0, eps, fusion_out)

    
#     label_sim_matrix = torch.mm(label, label.T).cuda()
#     label_sim_matrix[label_sim_matrix > 0] = 1

#     img_normalized_matrix_code = img_out / img_out.norm(dim=1, keepdim=True)
#     txt_normalized_matrix_code = txt_out / txt_out.norm(dim=1, keepdim=True)
#     fusion_normalized_matrix_code = fusion_out / fusion_out.norm(dim=1, keepdim=True)

#     img_sim_matrix = torch.mm(img_normalized_matrix_code, img_normalized_matrix_code.t())
#     txt_sim_matrix = torch.mm(txt_normalized_matrix_code, txt_normalized_matrix_code.t())
#     fusion_sim_matrix = torch.mm(fusion_normalized_matrix_code, fusion_normalized_matrix_code.t())
    
#     loss_sim_img = Loss_l2(img_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())
#     loss_sim_txt = Loss_l2(txt_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())
#     loss_sim_fusion = Loss_l2(fusion_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())

#     target = torch.ones(img_out.size(0)).cuda()  # 目标是1，表示希望两个向量尽可能相似
#     if args.fusion:
#         loss_align = criterion_cos(img_out, fusion_out.detach(), target) + criterion_cos(txt_out, fusion_out.detach(), target)
#         # loss_align = Loss_l2(img_out, fusion_out.detach()) + Loss_l2(txt_out, fusion_out.detach())
#     else:
#         loss_align = criterion_cos(img_out, txt_out, target)
#         # loss_align = Loss_l2(img_out, txt_out)

#     # 量化误差
#     loss_qua = Loss_l2(img_out - img_out.sign(), torch.zeros_like(img_out).cuda()) + \
#                Loss_l2(txt_out - txt_out.sign(), torch.zeros_like(txt_out).cuda()) + \
#                Loss_l2(fusion_out - fusion_out.sign(), torch.zeros_like(fusion_out).cuda())
    
#     if args.fusion:
#         loss = loss_sim_fusion * args.loss_fusion_sim_factor + \
#                 loss_sim_img * args.loss_modal_sim_factor + \
#                 loss_sim_txt * args.loss_modal_sim_factor + \
#                 loss_align * args.loss_align_factor  # + loss_qua * 1
#     else:
#         loss = loss_sim_fusion * 0 + \
#                 loss_sim_img * args.loss_modal_sim_factor + \
#                 loss_sim_txt * args.loss_modal_sim_factor + \
#                 loss_align * args.loss_align_factor  # + loss_qua * 1

#     loss_log = {'loss_sim_fusion': loss_sim_fusion.item(), 'loss_sim_img': loss_sim_img.item(), 'loss_sim_txt': loss_sim_txt.item(), \
#                 'loss_align': loss_align.item(), 'loss_qua': loss_qua.item(), 'loss_total': loss.item()}

#     return loss, loss_log

def new_cal_hash_loss(args, img_out, txt_out, fusion_img_out, fusion_txt_out, fusion_v, label):

    eps = torch.tensor(1e-5).cuda(non_blocking=True)

    # 相似度约束
    img_out = torch.where(img_out == 0, eps, img_out)
    txt_out = torch.where(txt_out == 0, eps, txt_out)
    fusion_v = torch.where(fusion_v == 0, eps, fusion_img_out)
    
    label_sim_matrix = torch.mm(label, label.T).cuda()
    label_sim_matrix[label_sim_matrix > 0] = 1

    img_normalized_matrix_code = img_out / img_out.norm(dim=1, keepdim=True)
    txt_normalized_matrix_code = txt_out / txt_out.norm(dim=1, keepdim=True)
    fusion_v_normalized_matrix_code = fusion_v / fusion_v.norm(dim=1, keepdim=True)

    # img_sim_matrix = torch.mm(img_normalized_matrix_code, img_normalized_matrix_code.t())
    # txt_sim_matrix = torch.mm(txt_normalized_matrix_code, txt_normalized_matrix_code.t())
    img_txt_sim_matrix = torch.mm(img_normalized_matrix_code, txt_normalized_matrix_code.t())
    fusion_v_sim_matrix = torch.mm(fusion_v_normalized_matrix_code, fusion_v_normalized_matrix_code.t())
    
    # loss_sim_img = Loss_l2(img_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())
    # loss_sim_txt = Loss_l2(txt_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())
    # loss_sim_modal = loss_sim_img + loss_sim_txt


    loss_align_sim_modal = Loss_l2(img_txt_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())

    target = torch.zeros_like(img_out).cuda()  # 目标是1，表示希望两个向量尽可能相似
    if args.fusion:
        loss_sim_fusion_v = Loss_l2(fusion_v_sim_matrix - label_sim_matrix, torch.zeros_like(label_sim_matrix).cuda())
        loss_reserve = Loss_l2(fusion_img_out - img_out.detach(), target) + Loss_l2(fusion_txt_out - txt_out.detach(), target)
        loss_distillation = Loss_l2(img_out - fusion_v.detach(), target) + Loss_l2(txt_out - fusion_v.detach(), target)
    else:        
        loss_align_modal = Loss_l2(img_out - txt_out, torch.zeros_like(img_out).cuda())

    # 量化误差
    loss_qua = Loss_l2(img_out - img_out.sign(), target) + \
               Loss_l2(txt_out - txt_out.sign(), target) + \
               Loss_l2(fusion_img_out - fusion_img_out.sign(), target) + \
               Loss_l2(fusion_txt_out - fusion_txt_out.sign(), target)
    
    if args.fusion:
        loss = loss_align_sim_modal * 1 + \
               loss_sim_fusion_v * 1 + \
               loss_reserve * 0 + \
               loss_distillation * 0.1
        
        loss_log = {'loss_align_sim_modal': loss_align_sim_modal.item(), 'loss_sim_fusion_v': loss_sim_fusion_v.item(), 'loss_reserve': loss_reserve.item(), \
                'loss_distillation': loss_distillation.item(), 'loss_total': loss.item()}
    else:
        loss = loss_align_sim_modal * 1 + \
               loss_align_modal * 1  # + loss_qua * 1

        loss_log = {'loss_sim_modal': loss_align_sim_modal.item(), 'loss_align_modal': loss_align_modal.item(), 'loss_total': loss.item()}

    assert not torch.isnan(loss).any() , "Loss is Nan !"

    return loss, loss_log


def integrat_loss_log(total_loss_log, loss_log):
    for k, v in loss_log.items():
        if k in total_loss_log:
            total_loss_log[k] += v
        else:
            total_loss_log[k] = v


# @snoop()
def train_one_epoch(args, img_model, txt_model, fusion_model, tr_loader, optimizer):

    epoch_loss_log = {}
    batch_loss_log = {}
    for img_path, img, txt, label in tqdm(tr_loader, disable=args.tqdm_disable):
        optimizer.zero_grad()
        if args.freeze_img_backbone:
            img = img.permute(1, 0, 2)
        img = img.cuda()
        txt = txt.cuda()
        label = label.cuda()
        hash_label = label

        with torch.amp.autocast('cuda'):
            img_out = img_model(img)
            txt_out = txt_model(txt)

            # fusion_img, fusion_txt, fusion_v = fusion_model(img_out[:-1], txt_out[:-1])
            fusion_v = fusion_model(img_out[:-1], txt_out[:-1])
            loss, batch_loss_log = cal_hash_loss(args, img_out[-1].mean(0), txt_out[-1].mean(0), fusion_v.mean(0), hash_label)
            # fusion_img_out, fusion_txt_out, fusion_v = fusion_model(img_out, txt_out)
            # loss, batch_loss_log = cal_hash_loss(args, img_out[-1].mean(0), txt_out[-1].mean(0), fusion_img_out.mean(0), fusion_txt_out.mean(0), fusion_v.mean(0), hash_label)
            
            loss.backward()
            optimizer.step()
        
        functional.reset_net(img_model)
        functional.reset_net(txt_model)
        functional.reset_net(fusion_model)

        integrat_loss_log(epoch_loss_log, batch_loss_log)

    wandb.log(epoch_loss_log)
   

def extract_flickr_feat(args, model, dataloader):
    img_path_l = list()
    img_feat_l = list()
    txt_l = list()
    label_l = list()

    cnt = 0

    with torch.no_grad():
        for img_path, img, bow, label in tqdm(dataloader):

            img = img.cuda()
            img_path_l.append(img_path)
            img_feat = model(img)
            img_feat_l.append(img_feat.cpu().numpy())
            txt_l.append(bow.numpy())
            label_l.append(label.numpy())

            functional.reset_net(model)
            # cnt += 1
            # if cnt > 2:
            #     break

        img_path = np.concatenate(img_path_l)
        img_feat = np.concatenate(img_feat_l, axis=1)
        img_feat = img_feat.transpose(1, 0, 2) # 转换为 B, T, C
        txt = np.concatenate(txt_l)
        label = np.concatenate(label_l)

    return img_path, img_feat, txt, label

def create_dataloader_for_head_energy(args):
    dataset = args.dataset
    test_feat_path =  f'/data/zhangzhen/CMR/data/feat/{dataset}/author_ckp_meta_former/test_feat_meta_former_T{args.img_T}.npy'

    test_feat = np.load(test_feat_path, allow_pickle=True).item()
    img_dataset = MyEnergyImageDataset(args, test_feat)
    txt_dataset = MyEnergyTextDataset(args, test_feat)

    img_dataloader = DataLoader(img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    txt_dataloader = DataLoader(txt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return img_dataloader, txt_dataloader


def create_dataloader_for_freeze_backbone(args):
    dataset = args.dataset
    T = args.img_T
    if args.img_backbone == 'resnet18':
        # train_feat_path = f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_spiking_resnet/train_feat_spiking_resnet18_T{T}.npy'
        # test_feat_path = f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_spiking_resnet/test_feat_spiking_resnet18_T{T}.npy'
    
        train_feat_path = f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_spiking_resnet_spike_feat/train_feat_spiking_resnet18_T{T}.npy'
        test_feat_path =  f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_spiking_resnet_spike_feat/test_feat_spiking_resnet18_T{T}.npy'
        # train_feat_path = f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_spiking_resnet_spike_feat_reset/train_feat_spiking_resnet18_T{T}.npy'
        # test_feat_path = f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_spiking_resnet_spike_feat_reset/test_feat_spiking_resnet18_T{T}.npy'

    elif args.img_backbone == 'meta_former':
        train_feat_path = f'/data/zhangzhen/CMR/data/feat/{dataset}/author_ckp_meta_former/train_feat_meta_former_T{T}.npy'
        test_feat_path =  f'/data/zhangzhen/CMR/data/feat/{dataset}/author_ckp_meta_former/test_feat_meta_former_T{T}.npy'
        # train_feat_path = f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_meta_former/train_feat_meta_former_T{T}.npy'
        # test_feat_path =  f'/data/zhangzhen/CMR/data/feat/{dataset}/my_pretrain_meta_former/test_feat_meta_former_T{T}.npy'
        pass
    train_feat = np.load(train_feat_path, allow_pickle=True).item()
    test_feat = np.load(test_feat_path, allow_pickle=True).item()

    tr_loader, te_loader, db_loader = create_common_dataloader(args, train_feat, test_feat, train_feat, img_dir='', cur_task='hash')

    return tr_loader, te_loader, db_loader


# def create_dataloader_for_freeze_backbone(args, img_backbone, dataloaders):
#     tr_loader, te_loader, db_loader = dataloaders

#     img_backbone = img_backbone.cuda()
#     img_backbone.eval()

#     all_new_sample_info = list()

#     for loader in [tr_loader, te_loader, db_loader]:
#         new_sample_info = dict()
#         img_path, img_feat, txt, label = extract_flickr_feat(args, img_backbone, loader)
#         new_sample_info['img_path'] = img_path
#         new_sample_info['img_feat'] = img_feat
#         new_sample_info['txt'] = txt
#         new_sample_info['label'] = label

#         all_new_sample_info.append(new_sample_info)

#     tr_loader, te_loader, db_loader = create_common_dataloader(args, all_new_sample_info[0], all_new_sample_info[1], all_new_sample_info[2], img_dir='', cur_task='hash')

#     return tr_loader, te_loader, db_loader

def freeze_bn(L):
    if isinstance(L, nn.BatchNorm1d):
        L.eval()

def train_hash(args):
    
    setup_seed(42)
    my_print("Start hash train...")
    my_print("Create dataloader...")

    if args.freeze_img_backbone:
        cur_task = 'extract_ferat'
    else:
        cur_task = 'hash'

    if args.dataset =='flickr':
        txt_dim = 1386
        label_dim = 24
    elif args.dataset == 'coco':
        txt_dim = 2000
        label_dim = 80
    elif args.dataset == 'nuswide':
        txt_dim = 1000
        label_dim = 21
    else:
        raise NotImplementedError
    
    if args.freeze_img_backbone:
        # 如果是锁定backbone，那么使用预训练权重提取一遍特征，然后重新制作数据集
        # img_model, txt_model, fusion_model = build_model(args, txt_dim=txt_dim, label_dim=label_dim, img_model_type='backbone')
        # lood_pretrain_ckp(args, img_model)
        # img_model.head = nn.Sequential()
        # img_backbone = img_model
        
        # my_print("Create new dataloader for freeze_img_backbone...")
        # tr_loader, te_loader, db_loader = create_dataloader_for_freeze_backbone(args, img_backbone, [tr_loader, te_loader, db_loader])
        
        # del img_model, img_backbone, txt_model
        # torch.cuda.empty_cache()
        tr_loader, te_loader, db_loader = create_dataloader_for_freeze_backbone(args)
        img_model, txt_model, fusion_model = build_model(args, txt_dim=txt_dim, label_dim=label_dim, img_model_type='head')

    else:
        train_info, test_info, db_info, img_dir = load_sample_info(args)
        tr_loader, te_loader, db_loader = create_common_dataloader(args, train_info, test_info, db_info, img_dir, cur_task) # 构造普通获取原图的dataloader
        img_model, txt_model, fusion_model = build_model(args, txt_dim=txt_dim, label_dim=label_dim, img_model_type='all')
        lood_pretrain_ckp(args, img_model)

    img_model_params = sum(p.numel() for p in img_model.parameters())
    my_print(f'Image model Total Params: {img_model_params:,}')

    txt_model_params = sum(p.numel() for p in txt_model.parameters())
    my_print(f"Txt model Total Params: {txt_model_params:,}")
    
    fusion_model_params = sum(p.numel() for p in fusion_model.parameters())
    my_print(f"Fusion model Total Params: {fusion_model_params:,}")

    img_model.cuda()
    txt_model.cuda()
    fusion_model.cuda()

    if args.freeze_img_backbone: # 此时的img_modal只有img_head
        params = [{'params': filter(lambda p: p.requires_grad, img_model.parameters()), 'lr': args.image_head_lr}] +\
                [{"params": filter(lambda p: p.requires_grad, txt_model.parameters()), "lr": args.txt_lr}]  +\
                [{"params": filter(lambda p: p.requires_grad, fusion_model.parameters()), "lr": args.fusion_lr}]
    else:
        params = [{'params': [param for name, param in img_model.named_parameters() if 'head' not in name and param.requires_grad], 'lr': args.img_backbone_lr}] +\
                [{"params": filter(lambda p: p.requires_grad, img_model.head.parameters()), "lr": args.image_head_lr}] +\
                [{"params": filter(lambda p: p.requires_grad, txt_model.parameters()), "lr": args.txt_lr}]  +\
                [{"params": filter(lambda p: p.requires_grad, fusion_model.parameters()), "lr": args.fusion_lr}]

    # optimizer = torch.optim.Adam(params=params)
    optimizer = torch.optim.SGD(params=params, weight_decay=args.weight_decay)
    if args.hash_scheduler_lr:
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    else:
        scheduler_lr = None

    best_score = 0
    best_score_epoch = 0
    for epoch in range(args.epoch):
        img_model.train()
        txt_model.train()
        fusion_model.train()

        if epoch > 0:
            img_model.apply(freeze_bn)
            txt_model.apply(freeze_bn)
            fusion_model.apply(freeze_bn)

        train_one_epoch(args, img_model, txt_model, fusion_model, tr_loader, optimizer)
        
        if scheduler_lr is not None:
            scheduler_lr.step()

        if epoch % args.eval_interval == 0:
            MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, HASH_CODE = validate(args, img_model, txt_model, fusion_model, te_loader, db_loader, topk=args.topk)
           
            state_dict = {'img_model': img_model.state_dict(), 'txt_model': txt_model.state_dict(), 'fusion_model': fusion_model.state_dict()}
            if MAP_I2T + MAP_T2I > best_score:
                best_score = MAP_I2T + MAP_T2I
                best_score_epoch = epoch
                Best_I2T, Best_T2I = MAP_I2T, MAP_T2I

                best_hash_code = HASH_CODE
                best_state_dict = state_dict
            
            if args.freeze_img_backbone:
                backbon_lr = 0.
                img_head_lr = optimizer.param_groups[0]["lr"]
                txt_lr = optimizer.param_groups[1]["lr"]
            else:
                backbon_lr = optimizer.param_groups[0]["lr"]
                img_head_lr = optimizer.param_groups[1]["lr"]
                txt_lr = optimizer.param_groups[2]["lr"]
            result = "Data:{:s}, T={:d}, Bit={:d}, Epoch:{:3d}, I2T: {:.3f}, T2I:{:.3f} --- I2I:{:.3f}, T2T:{:.3f} Lr:{:.5f}-{:.5f}-{:.5f} ### BestEpoch:{:3d}, Best: I2T:{:.3f}, T2I:{:.3f}"\
                .format(args.dataset, args.img_T, args.bit_len, epoch, MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, backbon_lr, txt_lr, img_head_lr, best_score_epoch, Best_I2T, Best_T2I)
            my_print(result)

    # save_info(args, best_state_dict, result, best_hash_code)


def save_info(args, state_dict, result, hash_code):
    save_dir = f'/data/zhangzhen/logs/{args.img_backbone}/{args.dataset}/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # 存模型
    file_name = f'{args.img_T}-{args.bit_len}-{args.fusion}.pth'
    file_path = os.path.join(save_dir, file_name)
    torch.save(state_dict, file_path)

    # 存哈希码
    te_BI, te_BT, te_L, db_BI, db_BT, db_L = hash_code
    te_BI, te_BT, te_L, db_BI, db_BT, db_L = te_BI.cpu().numpy(), te_BT.cpu().numpy(), te_L.cpu().numpy(), db_BI.cpu().numpy(), db_BT.cpu().numpy(), db_L.cpu().numpy()

    file_name = f'{args.img_T}-{args.bit_len}-{args.fusion}.mat'
    file_path = os.path.join(save_dir, file_name)
    data = dict()
    data['te_BI'] = te_BI
    data['te_BT'] = te_BT
    data['te_L'] = te_L
    data['db_BI'] = db_BI
    data['db_BT'] = db_BT
    data['db_L'] = db_L
    savemat(file_path, data)

    # 存性能结果
    log_path = os.path.join(save_dir, 'log.txt')
    with open(log_path, 'a', encoding='utf-8') as file:
        var_args = f'T={args.img_T}, bit={args.bit_len}, fusion={args.fusion}\n'
        file.write(var_args)
        file.write(result + '\n')
        

def pretrain(args):
    
    setup_seed(42)
    # if args.arch == "ANN":
    #     assert False, 'Ann dose not need to train'
    my_print("Start pretrain ...")

    train_info, test_info, db_info, img_dir = load_sample_info(args)
    tr_loader, _, _ = create_common_dataloader(args, train_info, test_info, db_info, img_dir, cur_task='pretrain')
    txt_dim = tr_loader.dataset.get_txt_dim()
    label_dim = tr_loader.dataset.get_label_dim()

    img_model, _, _ = build_model(args, txt_dim=txt_dim, label_dim=label_dim, img_model_type='backbone')
    img_model.cuda()
    
    params = [{"params": filter(lambda p: p.requires_grad, img_model.parameters()), "lr": args.pretrain_lr}]
    optimizer = torch.optim.SGD(params=params, weight_decay=args.weight_decay)

    if args.pretrain_scheduler_lr:
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    else:
        scheduler_lr = None

    for epoch in range(args.epoch):

        epoch_loss_log = dict()
        epoch_loss_log["loss_cls"] = 0

        my_print("Epoch: %3d" % epoch)
        for img_path, img, bow, label in tqdm(tr_loader, disable=args.tqdm_disable):

            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()

            with torch.amp.autocast('cuda'):
                img_out = img_model(img)
                assert img_out.dim() == 3
                loss = 0
                for img_out_one_step in img_out:
                    loss = loss + criterion_CE(img_out_one_step, label)
                loss = loss / len(img_out)
                loss.backward()
                optimizer.step()

            functional.reset_net(img_model)
            epoch_loss_log["loss_cls"] += loss.item()
        
        if scheduler_lr is not None:
            scheduler_lr.step()

        if epoch == 0 or (epoch + 1) % 10 == 0: # 每10次存1个
            if not os.path.exists(args.cls_ckp_dir):
                os.makedirs(args.cls_ckp_dir)
            ckp_path = os.path.join(args.cls_ckp_dir, "epoch_{}.pth".format(epoch))
            save_data = dict()
            assert not os.path.exists(ckp_path)
            state_dict = img_model.state_dict()
            save_data['args'] = str(args)
            save_data['ckp'] = state_dict
            torch.save(save_data, ckp_path)

        loss_log = {"loss_pretrain": epoch_loss_log}
        wandb.log(loss_log)


def cal_energy(args):

    if args.dataset == 'flickr':
        label_dim = 24
        txt_dim = 1386
    elif args.dataset == 'coco':
        label_dim = 80
        txt_dim = 2000
    elif args.dataset == 'nuswide':
        label_dim = 21
        txt_dim = 1000


    train_info, test_info, db_info, img_dir = load_sample_info(args)
    img_loader, txt_loader = create_energy_dataloader(args, test_info, img_dir)
    # tr_loader, te_loader, db_loader = create_dataloader_for_freeze_backbone(args)
    # img_loader, txt_loader = create_dataloader_for_head_energy(args)
    img_model = build_img_backbone(args, label_dim)
    lood_pretrain_ckp(args, img_model)

    img_head = ImageHashHead(args, 512, args.bit_len)
    txt_moel = TxtModel(args, txt_dim)

    # p = os.path.join(f'/data/zhangzhen/logs/my_pretrain_resnet/{args.dataset}/', f'{args.img_T}-{args.bit_len}-True-map.pth')
    p = os.path.join(f'/data/zhangzhen/logs/resnet18/{args.dataset}/', f'{args.img_T}-{args.bit_len}-True-ours.pth')
    assert os.path.exists(p)

    head_stat_dict = torch.load(p)
    img_head.load_state_dict(head_stat_dict['img_model'])
    txt_moel.load_state_dict(head_stat_dict['txt_model'])

    img_model.head = img_head

    (img_tot, img_acs, img_macs, img_fire_rate), img_pamrams = get_model_complexity_info(img_model, None, img_loader, as_strings=False, print_per_layer_stat=True, verbose=False, batch_dim_idx=0)
    img_ac_energy = img_acs * 0.9 * 1e-9
    img_mac_energy = img_macs * 4.6 * 1e-9
    (txt_tot, txt_acs, txt_macs, txt_fire_rate), txt_pamrams = get_model_complexity_info(txt_moel, None, txt_loader, as_strings=False, print_per_layer_stat=True, verbose=False, batch_dim_idx=1)
    txt_ac_energy = txt_acs * 0.9 * 1e-9
    txt_mac_energy = txt_macs * 4.6 * 1e-9

    print(f'img_acs:{img_acs:.4g},  img_macs:{img_macs:.4g}, -----img_ac_energy:{img_ac_energy:.4g},  img_mac_rnergy{img_mac_energy:.4g}')
    print(f'txt_acs:{txt_acs:.4g},  txt_macs:{txt_macs:.4g}, -----txt_ac_energy:{txt_ac_energy:.4g},  txt_mac_rnergy{txt_mac_energy:.4g}')

    total_energy = img_ac_energy + img_mac_energy + txt_ac_energy + txt_mac_energy
    print(f'total_energy:{total_energy:.4g}')


    pass



def main():
    wandb.init(project="snn_v5")
    args = parse_args()
    my_print(args)
    if args.task == "pretrain":
        pretrain(args)

    elif args.task == "hash":
        train_hash(args)

    elif args.task == "full":
        pretrain(args)
        train_hash(args)

    elif args.task == "cal_energy":
        cal_energy(args)
    else:            
        assert False, "Unknown task"

if __name__ == "__main__":
    main()

