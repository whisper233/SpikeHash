import torch
import torch.nn as nn

from spikingjelly.activation_based import functional as snn_functional
from spikingjelly.activation_based import layer as snn_layer
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

   

class TransformFusionBlock(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        
        super().__init__()
        thresh_learnable = False

        self.img_q_block = snn_layer.MultiStepContainer(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
        self.img_q_lif = LIF_TET(thresh_learnable=thresh_learnable)
        self.img_k_block = snn_layer.MultiStepContainer(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
        self.img_k_lif = LIF_TET(thresh_learnable=thresh_learnable)

        self.txt_q_block = snn_layer.MultiStepContainer(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
        self.txt_q_lif = LIF_TET(thresh_learnable=thresh_learnable)
        self.txt_k_block = snn_layer.MultiStepContainer(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))
        self.txt_k_lif = LIF_TET(thresh_learnable=thresh_learnable)

        self.v_block = snn_layer.MultiStepContainer(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))

        self.fusion_lif = LIF_TET()

    def forward(self, img, txt, last_v=None): # shape T, N, C

        img_q = self.img_q_block(img)
        img_q_spike = self.img_q_lif(img_q)

        txt_q = self.txt_q_block(txt)
        txt_q_spike = self.txt_q_lif(txt_q)
        if last_v is None:
            # first layer
            fusion = torch.stack((img, txt), dim=0).max(dim=0).values
        else: 
            fusion = torch.stack((img, txt, last_v), dim=0).max(dim=0).values

        img_k = self.img_k_block(fusion)
        img_k_spike = self.img_k_lif(img_k)

        txt_k = self.txt_k_block(fusion)
        txt_k_spike = self.txt_k_lif(txt_k)

        v = self.v_block(fusion)

        img_fusion = img_q_spike * img_k_spike * v
        txt_fusion = txt_q_spike * txt_k_spike * v

        res = img_fusion + txt_fusion + v

        res = self.fusion_lif(res)

        return res


class DenseNet(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        self.layer_num = len(module_list)

    def forward(self, x):
        save = x
        ret = list()

        for i in range(self.layer_num):
            x = self.module_list[i](save)
            if i == 0:
                save = x
            elif i != self.layer_num - 1:
                save = save + x
            else:
                pass

            ret.append(x)
            
        return ret

    

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIF_TET(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, thresh_learnable=False):
        super(LIF_TET, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        if thresh_learnable:
            self.thresh = nn.Parameter(torch.tensor(thresh))
        else:
            self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * self.tau + x[t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=0)


# HASH 
class SnnMlpLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        
        self.bn = nn.BatchNorm1d(out_dim)

        self.act = LIF_TET()

    def forward(self, x):
        T,B,C = x.shape

        assert C == self.in_dim, "SnnMlpLayer input dimension error!"

        out = snn_functional.multi_step_forward(x, (self.linear, self.bn))

        out = self.act(out)

        return out


class AnnMlpLayer(nn.Module):
    def __init__(self, args, in_dim, out_dim, act=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        # self.act = nn.Sigmoid()
        if act is None:
            self.act = nn.ReLU()
        else:
            self.act = act
        # self.act = nn.ReLU()

    def forward(self, x):
        B,C = x.shape

        assert C == self.in_dim, "AnnMlpLayer input dimension error!"
        x = self.linear(x)
        x = self.bn(x)
        output = self.act(x)
        return output


# class ClassifyHead(nn.Module):
#     def __init__(self, args, in_dim, class_num):
#         super().__init__()
#         self.args = args
#         self.in_dim = in_dim
#         self.linear = nn.Linear(in_dim, class_num)
#         # self.softmax = nn.Softmax(dim=1)
#         # self.act = nn.Sigmoid()

#     def forward(self, x):
#         if self.args.SNN_hash_layer:
#             out = snn_functional.multi_step_forward(x, (self.linear, self.act))
#         else:
#             out = self.linear(x)
#             # out = self.act(out)
        
#         return out


def Snn2Ann(module):
    if isinstance(module, nn.Module):
        for name, child in module.named_children():
            if isinstance(child, MultiStepLIFNode):
                # new_m = nn.Sigmoid()
                # new_m = nn.Tanh()
                new_m = nn.ReLU()
                setattr(module, name, new_m)
            else:
                Snn2Ann(child)
