import os
import random
import numpy as np
import torch
from spikingjelly.clock_driven.model import spiking_resnet
from spikingjelly.clock_driven.model import sew_resnet
from spikingjelly.clock_driven import surrogate, functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from torchvision.models import resnet as ann_resnet
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from syops import get_model_complexity_info

from meta_spikingformer import metaspikformer_8_512
from hash_model import lood_pretrain_ckp, custom_load_state_dict
from load_data import build_transform, load_sample_info
from print_tools import my_print


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Args:
    def __init__(self):
        self.T = 2
        self.dataset = 'flickr'
        self.input_size = 224
        self.sample_num = 1000
        self.bit_len = 32
        self.img_backbone = 'meta_former'
        self.cls_ckp_dir = '/data/zhangzhen/ckp/12.5-1/T2/meta_former'

class MyDatasetEnergy(Dataset):
    def __init__(self, args, data_type, sample_info, img_dir, transform=None): # data_type: 'src_img' or 'feat'
        self.args = args
        self.img_path = sample_info['img_path']
        self.img_dir = img_dir
        self.transform = transform
        self.img_type = data_type
        
        if data_type == 'feat':
            self.feat = torch.randint(0, 2, (args.sample_num, args.T, 512))
        elif data_type == 'src_img':
            pass
        else:
            raise NotImplementedError

    def __len__(self):
        return self.args.sample_num

    def __getitem__(self, index):
        img_path = self.img_path[index]
        if self.img_type == 'feat':
            data = self.feat[index]

        elif self.img_type == 'src_img':
            abs_img_path = os.path.join(self.img_dir, img_path)
            pil_img = Image.open(abs_img_path).convert("RGB") # TODO
            data = self.transform(pil_img)
            # data = data.unsqueeze(0).repeat(self.args.T, 1, 1, 1)

            pil_img.close()

        else:
            raise NotImplementedError

        return data, index
    

class TempDataset(Dataset):
    def __init__(self, num=100):
        self.num = num
        pass

    def __getitem__(self, index):
        return torch.ones(3, 224, 224), index
    
    def __len__(self):
        return self.num

def cal_energy(img_model, input_size, dataloader):
    # import torch
    from spikingjelly.activation_based import surrogate, neuron, functional
    # from spikingjelly.activation_based.model import spiking_resnet
    from syops import get_model_complexity_info

    # dataloader = ...
    with torch.cuda.device(0):
        net = spiking_resnet.spiking_resnet18(pretrained=True, spiking_neuron=neuron.IFNode, 
                surrogate_function=surrogate.ATan(), detach_reset=True)
        ops, params = get_model_complexity_info(net, (3, 224, 224), dataloader, as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity ACs:', acs))
        # print('{:<30}  {:<8}'.format('Computational complexity MACs:', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        return ops, params


def temp_setup_seed(seed=42):
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


def build_model_energy(args):
    # 6层的block 3 meta-spikingformer
    my_print("loading META-FORMER checkpoint ! ! !")
    img_model = metaspikformer_8_512(T=args.T, num_classes=1000)
    ckp_path = "/data/zhangzhen/ckp/meta-former-imagenet/55M_kd.pth"
    # ckp_path = "/data/zhangzhen/ckp/meta-former-imagenet/55M_kd_T4.pth"
    ckp = torch.load(ckp_path)
    # my_print(ckp.keys())
    weight = ckp['model']
    custom_load_state_dict(img_model, weight)

    # 2层的block 3 meta-spikingformer
    # img_model = metaspikformer_8_512(T=args.T, num_classes=24)
    # lood_pretrain_ckp(args, img_model)
    
    return img_model

def swap_01_dim(x): # 默认当前只有两个参数
    l1 = list()
    l2 = list()

    for sample in x:
        l1.append(sample[0])
        l2.append(sample[1])

    x1 = torch.stack(l1)
    x2 = torch.tensor(l2)

    y = torch.transpose(x1, 0, 1)

    return y, x2

def build_dataloader_energy(args, data_type):
    
    train_info, test_info, db_info, img_dir = load_sample_info(args)
    img_transform = build_transform(is_train=False, args=args)
    train_dataset = MyDatasetEnergy(args, data_type=data_type, sample_info=train_info, img_dir=img_dir, transform=img_transform)

    # return DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=swap_01_dim)
    return DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=None)


def main():
    temp_setup_seed()

    args = Args()

    # TET
    # sys.path.append('/home/zhangzhen/git/temporal_efficient_training/models')
    # from resnet_models import resnet19 as TET_model
    # model = TET_model(num_classes=10)
    # model = spiking_resnet.multi_step_spiking_resnet18(pretrained=True, T=args.T, multi_step_neuron=MultiStepLIFNode, surrogate_function=surrogate.ATan(), detach_reset=True, tau=2.0, backend="cupy")
    # model = MyResNet(args, num_classes=24, T=args.T)
    # model = metaspikformer_8_512(T=args.T, num_classes=24)
    # lood_pretrain_ckp(args, model)
    model = ann_resnet.resnet18(weights=ann_resnet.ResNet18_Weights.IMAGENET1K_V1)
    
    # model = sew_resnet.multi_step_sew_resnet18(num_classes=1000, T=1, multi_step_neuron=MultiStepLIFNode, cnf='ADD', tau=2.0, detach_reset=True, backend="cupy")

    # model = resnet.resnet18(pretrained=True)
    # model = resnet.resnet50(pretrained=True)

    # dataloader = DataLoader(TempDataset(1000), batch_size=16, shuffle=False, num_workers=0)

    # model = build_model_energy(args)
    # model = ImageHashHead(args, in_dim=512, bit_len=args.bit_len)
    
    # dataloader = build_dataloader_energy(args, data_type='src_img')

    # ops, params = get_model_complexity_info(model, (3, 224, 224), dataloader, as_strings=False,
    #                                         print_per_layer_stat=True, verbose=False)

    model.cuda()
    ops, params = get_model_complexity_info(model, (3, 224, 224), None, as_strings=False,
                                            print_per_layer_stat=True, verbose=False, batch_dim_idx=0)
    
    print(params)
    print(ops)

    ac_energy = ops[1] * 0.9 / (1e9)
    mac_energy = ops[2] * 4.6 / (1e9)

    tot = ac_energy + mac_energy

    print(f'ac_energy {ac_energy:.3f} mJ, mac_energy {mac_energy:.3f} mJ, tot_energy: {tot:.3f} mJ')

    raw_ac_energy = (ops[0] - ops[2]) * 0.5 * 0.9 / (1e9) # 对于未训练的SNN，由于发射率不准确，暂时考虑为0.5
    tot = raw_ac_energy + mac_energy
    print(f'For raw SNN, raw_ac_energy {raw_ac_energy:.3f} mJ, mac_energy {mac_energy:.3f} mJ, tot_energy: {tot:.3f} mJ')

if __name__ == "__main__":
    main()
