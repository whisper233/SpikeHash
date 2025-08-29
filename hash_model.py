import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet18 as ANN_resnet18
from spikingjelly.clock_driven.neuron import LIFNode, MultiStepLIFNode
from spikingjelly.activation_based import functional as snn_functional
from spikingjelly.activation_based import layer as snn_layer
from spikingjelly.clock_driven.model.spiking_resnet import multi_step_spiking_resnet18 as SNN_resnet18
from spikingjelly.clock_driven.model.spiking_resnet import multi_step_spiking_resnet34 as SNN_resnet34
from spikingjelly.clock_driven.model.spiking_resnet import multi_step_spiking_resnet50 as SNN_resnet50
from spikingjelly.clock_driven.model.sew_resnet import multi_step_sew_resnet18 as SNN_sew_resnet18
from spikingjelly.clock_driven.model.sew_resnet import multi_step_sew_resnet50 as SNN_sew_resnet50
import pickle as pkl

from base_model import SnnMlpLayer, LIF_TET, TransformFusionBlock
from meta_spikingformer import metaspikformer_8_512
from print_tools import my_print
# from functorch import vmap
# from torch import vmap
# surrogate_func = surrogate.ATan()


class ImageHashHead(nn.Module):
    def __init__(self, args, in_dim=640, bit_len=32):
        super().__init__()
        self.args = args

        self.layers = nn.Sequential(SnnMlpLayer(in_dim, 4096),
                                    SnnMlpLayer(4096, 4096),
                                    SnnMlpLayer(4096, 4096),
                                    snn_layer.MultiStepContainer(nn.Linear(4096, bit_len)))
                                    # snn_layer.MultiStepContainer(nn.Linear(4096, bit_len), nn.BatchNorm1d(bit_len), nn.Tanh()))
       
    def forward(self, x):
        ret = list()
        ret.append(x)
        out = x

        for layer in self.layers:
            out = layer(out)
            ret.append(out)

        if self.training:
            return ret
        else:
            return ret[-1]


class TxtHashHead(nn.Module):
    def __init__(self, args, in_dim):
        super().__init__()
        self.args = args
        bit_len = args.bit_len
        self.T = args.txt_T
        
        self.layers = nn.Sequential(SnnMlpLayer(in_dim, 4096),
                            SnnMlpLayer(4096, 4096),
                            SnnMlpLayer(4096, 4096),
                            snn_layer.MultiStepContainer(nn.Linear(4096, bit_len)))
                            # snn_layer.MultiStepContainer(nn.Linear(4096, bit_len), nn.BatchNorm1d(bit_len), nn.Tanh()))

    def forward(self, x):
        ret = list()
        
        out = (x.unsqueeze(0)).repeat(self.T, 1, 1)
        ret.append(x)
        for layer in self.layers:
            out = layer(out)
            ret.append(out)

        if self.training:
            return ret
        else:
            return ret[-1]

def init_embedding_layer(input_dim, dset_name):      

    Embedding = nn.Linear(input_dim, 300)
    init_weights = None
    if dset_name.lower() == 'nuswide':
        init_weights = pkl.load(open('/home/zhangzhen/git/HMAH/models/nuswide_weights.pkl', 'rb'))['weights'].T
    elif dset_name.lower() == 'flickr':
        init_weights = pkl.load(open('/home/zhangzhen/git/HMAH/models/flickr_weights.pkl', 'rb'))['weights'].T
    elif dset_name.lower() == 'coco':
        init_weights = pkl.load(open('/home/zhangzhen/git/HMAH/models/coco_weights.pkl', 'rb'))['weights'].T
    else:
        raise NotImplementedError
    
    Embedding.weight = nn.Parameter(torch.Tensor(init_weights))
    Embedding.weight.requires_grad = False
    return Embedding

class TxtModel(nn.Module):
    def __init__(self, args, in_dim):
        super().__init__()
        # self.embedding_layer = init_embedding_layer(in_dim, args.dataset)
        # self.txt_head = TxtHashHead(args, 300)
        
        self.embedding_layer = None
        self.txt_head = TxtHashHead(args, in_dim)

    def forward(self, x):
        if self.embedding_layer is not None:
            x = self.embedding_layer(x)
        out = self.txt_head(x)

        return out



def find_latest_checkpoint(ckp_dir, cls_ckp_specific_epoch):
    if cls_ckp_specific_epoch is not None:
        ckp_path = os.path.join(ckp_dir, f'epoch_{cls_ckp_specific_epoch}.pth')
        my_print('Specific epoch %d', cls_ckp_specific_epoch)
        if not os.path.exists(ckp_path):
            raise FileNotFoundError(f'{ckp_path} Not found!')
        return ckp_path

    # 获取文件夹中的所有文件
    files = os.listdir(ckp_dir)

    # 使用正则表达式匹配文件名中的数字部分
    pattern = re.compile(r'epoch_(\d+)\.pth')

    # 找到数字最大的文件
    max_epoch = -1
    max_file = None

    for file in files:
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                max_file = file
    abs_path = os.path.join(ckp_dir, max_file)
    return abs_path


class MyResNet(nn.Module):
    def __init__(self, args, num_classes, T=1):
        super().__init__()

        neuron_args = dict()
        neuron = LIF_TET
        if args.img_backbone == "resnet18":
            self.backbone = SNN_resnet18(num_classes=num_classes, T=T, multi_step_neuron=neuron,  **neuron_args)
            img_hash_head_in_dim = 512
        elif args.img_backbone == "resnet50":
            img_hash_head_in_dim = 2048
            self.backbone = SNN_resnet50(num_classes=num_classes, T=T, multi_step_neuron=neuron, **neuron_args)
            
        # elif args.img_backbone == "sew_resnet18":
        #     self.backbone = SNN_sew_resnet18(num_classes=num_classes, T=T, multi_step_neuron=MultiStepLIFNode, cnf='ADD', tau=2.0, detach_reset=True, backend="cupy")
        #     img_hash_head_in_dim = 512
        # elif args.img_backbone == "sew_resnet50":
        #     self.backbone = SNN_sew_resnet50(num_classes=num_classes, T=T, multi_step_neuron=MultiStepLIFNode, cnf='ADD', tau=2.0, detach_reset=True, backend="cupy")
        #     img_hash_head_in_dim = 2048
        else:
            raise NotImplementedError
        
        self.backbone.fc = nn.Sequential()
        self.pre_head = snn_layer.MultiStepContainer(nn.Linear(img_hash_head_in_dim, img_hash_head_in_dim), nn.BatchNorm1d(img_hash_head_in_dim))
        self.pre_lif = LIF_TET()
        self.head = snn_layer.MultiStepContainer(nn.Linear(img_hash_head_in_dim, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.pre_head(x)
        x = self.pre_lif(x)
        x = self.head(x)

        return x


class SpikeFusionBlock(nn.Module):
    # mode: map/add_bn/add_scale/concat
    def __init__(self, args):
        super().__init__()
        self.blocks = list()

        # if self.mode == 'map':
        #     assert all(x == in_dims[0] for x in in_dims[1:])
        #     # out_dim = in_dims[0]
        #     pass

        # elif self.mode == 'add_bn':
        #     assert all(x == in_dims[0] for x in in_dims[1:])
        #     out_dim = in_dims[0]
        #     self.bn = nn.BatchNorm1d(out_dim)
        #     self.act = LIF_TET()

        # elif self.mode == 'add_scale':
        #     assert all(x == in_dims[0] for x in in_dims[1:])
        #     out_dim = in_dims[0]
        #     self.scale = nn.Parameter(torch.randn(out_dim))
        #     self.act = LIF_TET()

        # elif self.mode == 'concat':
        #     # out_dim = sum(in_dims)
        #     pass

        # else:
        #     raise NotImplementedError(self.mode)
        
        # self.out_dim = out_dim

    def forward(self, l:list):
        x = torch.stack(l, dim=2)
        #  T * B * branch * C

        # if self.mode == 'map':
        out = x.max(dim=2)[0]
        # elif self.mode == 'add_bn':
        #     out = x.sum(dim=2)
        #     out = snn_functional.multi_step_forward(out, (self.bn))
        #     out = self.act(out)

        # elif self.mode == 'add_scale':
        #     out = x.sum(dim=2) 
        #     out = out * self.scale
        #     out = self.act(out)

        # elif self.mode == 'concat':
        #     T, B, branch, C = x.shape
        #     out = x.reshape(T, B, -1)
        # else:
        #     raise NotImplementedError

        return out # T, B, C


class MHHLHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_1 = snn_layer.MultiStepContainer(nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024))
        self.lif = LIF_TET(thresh_learnable=True)
        self.linear_2 = snn_layer.MultiStepContainer(nn.Linear(1024, out_dim))

    def forward(self, x):
        out = self.linear_1(x)
        out = self.lif(out)
        out = self.linear_2(out)

        return out


class MHHL(nn.Module):
    def __init__(self, args, in_dim):
        super().__init__()

        self.args = args
        self.head_num = args.bit_len
        self.heads = nn.ModuleList()
        
        assert in_dim % self.head_num == 0, 'Head num Error!'

        self.embedding_dim = int(in_dim / self.head_num)

        for i in range(self.head_num):
            head = MHHLHead(self.embedding_dim, 1)
            self.heads.append(head)
        

    def forward(self, x):
        # x shape T, B, 4096
        original_shape = x.shape
        xs = x.reshape(*original_shape[:-1], self.head_num, self.embedding_dim)

        l = list()
        for i in range(self.head_num):
            head = self.heads[i]

            one_bit = head(xs[:,:,i,:]) # T, B, head_num, 1
            l.append(one_bit)

        res = torch.cat(l, dim=-1)

        return res
        # split_x = torch.split(x, x.size(-1) // self.head_num, dim=-1)
        # split_x = torch.stack(split_x, dim=0)

        # res = vmap(lambda submodule, x: submodule(x))(self.heads, split_x)

        # return torch.cat(res, dim=-1)


        

class NewSpikeFusion(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super().__init__()

        self.block1 = TransformFusionBlock(args, in_dim, in_dim)
        self.block2 = TransformFusionBlock(args, in_dim, in_dim)
        self.block3 = TransformFusionBlock(args, in_dim, in_dim)

        # self.img_linear = snn_layer.MultiStepContainer(nn.Linear(in_dim, in_dim))
        # self.txt_linear = snn_layer.MultiStepContainer(nn.Linear(in_dim, in_dim))

        # self.img_hash_layer = snn_layer.MultiStepContainer(nn.Linear(in_dim, out_dim))
        # self.txt_hash_layer = snn_layer.MultiStepContainer(nn.Linear(in_dim, out_dim))
        # self.fusion_hash_layer = MHHL(args, 4096)
        self.fusion_hash_layer = snn_layer.MultiStepContainer(nn.Linear(in_dim, out_dim))

        # self.fusion_blocks = nn.Sequential(
        #     TransformFusionBlock(in_dim, out_dim)
        # )

    def forward(self, img, txt):
        # img_out, txt_out = self.fusion_blocks(img, txt)
        fusion = self.block1(img[-3], txt[-3])
        fusion = self.block2(img[-2], txt[-2], fusion)
        fusion = self.block3(img[-1], txt[-1], fusion)

        # img_out = self.img_linear(img_out)
        # txt_out = self.txt_linear(txt_out)

        fusion_out = self.fusion_hash_layer(fusion)

        return fusion_out

class SpikeFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_dim = 4096
        self.fusion_modules = nn.Sequential(SpikeFusionBlock(args),
                                            SpikeFusionBlock(args),
                                            SpikeFusionBlock(args))

        self.blocks =  nn.Sequential(SnnMlpLayer(hidden_dim, hidden_dim),
                                    SnnMlpLayer(hidden_dim, hidden_dim),
                                    nn.Linear(hidden_dim, args.bit_len))

        assert len(self.fusion_modules) == len(self.blocks)

    def forward(self, img_feats:list, txt_feats:list):
        out = None
        layer_num = len(self.fusion_modules)

        save = []
        for i, fusion_block in enumerate(zip(self.fusion_modules, self.blocks)):
            fusion, block = fusion_block
            if i == 0:
                out = fusion([img_feats[i], txt_feats[i]] + save)
            else:
                out = fusion(save)

            out = block(out)
            save = [out]

        return out


def custom_load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_dict:
            model_dict[name].copy_(param)
        else:
            my_print(f"Warning: {name} not found in model ! ! !")
    model.load_state_dict(model_dict)


def build_model_ori(args, txt_dim):
    my_print("loading META-FORMER checkpoint ! ! !")
    img_model = metaspikformer_8_512(T=args.T, num_classes=1000)
    ckp_path = "/data/zhangzhen/ckp/meta-former-imagenet/55M_kd.pth"
    # ckp_path = "/data/zhangzhen/ckp/meta-former-imagenet/55M_kd_T4.pth"
    ckp = torch.load(ckp_path)
    # my_print(ckp.keys())
    weight = ckp['model']
    custom_load_state_dict(img_model, weight)
    
    if args.freeze_img_backbone:
        for param in img_model.parameters():
            param.requires_grad = False

    img_model.head = ImageHashHead(args.T, in_dim=640, bit_len=args.bit_len)
    txt_model = TxtModel(args, in_dim=txt_dim)

    return img_model, txt_model


def build_img_backbone(args, label_dim):
    back_bone = args.img_backbone.lower()
    if 'resnet' in back_bone:
        img_model = MyResNet(args, T=args.img_T, num_classes=label_dim)
    elif back_bone == 'meta_former':
        img_model = metaspikformer_8_512(T=args.img_T, num_classes=label_dim)
    else:
        raise NotImplementedError

    return img_model


def lood_pretrain_ckp(args, model):
    ckp_path = find_latest_checkpoint(args.cls_ckp_dir, args.cls_ckp_specific_epoch)
    ckp = torch.load(ckp_path, weights_only=True)
    ckp_args = ckp["args"]
    ckp_param = ckp["ckp"]

    if args.img_backbone not in ckp_args:
        my_print('check point img backbone not match. current: ', args.img_backbone, ' ckp: ', ckp_args.img_backbone)
        raise RuntimeError

    my_print("loading ", ckp_path, " ...")
    custom_load_state_dict(model, ckp_param)
    my_print("loading done ")
    my_print("pretrain args: ", ckp_args)


def build_model(args, txt_dim, label_dim, img_model_type): # img_type: backbone/head/all
    
    if 'resnet18' in args.img_backbone:
        img_hash_head_in_dim = 512
    elif 'resnet50' in args.img_backbone:
        img_hash_head_in_dim = 2048
    elif args.img_backbone == 'meta_former':
        img_hash_head_in_dim = 640
    else:
        raise NotImplementedError
    
    if img_model_type == 'backbone':
        img_model = build_img_backbone(args, label_dim)
    elif img_model_type == 'head':
        img_model = ImageHashHead(args, in_dim=img_hash_head_in_dim, bit_len=args.bit_len)
    elif img_model_type == 'all':
        img_model = build_img_backbone(args, label_dim)
        img_hash_head = ImageHashHead(args, in_dim=img_hash_head_in_dim, bit_len=args.bit_len)

        img_model.head = img_hash_head
    else:
        raise NotImplementedError
    
    txt_model = TxtModel(args, in_dim=txt_dim)

    # fusion_model = SpikeFusion(args)

    fusion_model = NewSpikeFusion(args, 4096, args.bit_len)
    

    return img_model, txt_model, fusion_model


if __name__ == "__main__":
    a = [1,2,3]
    b = ['a', 'b', 'c']

    for i in zip(a, b):
        print(i)

    vit = models.vit_b_16(pretrained=True)
    # vit.cuda()
    # v = nn.Parameter(torch.ones(1))

    # optm = torch.optim.SGD(params=[{'params':v, 'lr':0.1}])

    # # v = 1.
    # # lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy", v_threshold=v).cuda()
    # lif = LIF_TET(thresh=v).cuda()
    
    # x = torch.randn(1,5).cuda()

    # for i in range(10):

    #     y = lif(x)

    #     loss = y.sum()
    #     loss.backward()
    #     optm.step()

    # param_num = sum(p.numel() for p in lif.parameters())

    # print(f'param num: {param_num}')

    # res = find_latest_checkpoint('/data/zhangzhen/ckp/flickr_class_ckp_T1')
    # print(res)
    # model = SNN_resnet19(num_classes=10)
    # model = SNN_resnet18(pretrained=True, T=4, multi_step_neuron=MultiStepLIFNode, tau=2.0, detach_reset=True, backend="cupy")
    # model = SNN_vgg11(pretrained=True, T=4, multi_step_neuron=MultiStepLIFNode, tau=2.0, detach_reset=True, backend="cupy")
    # model = SNN_resnet34(multi_step_neuron=MultiStepLIFNode, tau=2.0, detach_reset=True, backend="cupy").cuda()
    # x = torch.randn(32, 4, 3, 224, 224).cuda()
    # y = model(x)

    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}")
    #     print(f"Parameter type: {type(param)}")
    #     print(f"Parameter shape: {param.shape}")
    #     print()
    # model.cuda()
    # pass
