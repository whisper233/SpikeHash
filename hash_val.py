import torch
import torch.distributed as dist
from spikingjelly.clock_driven import functional
from tqdm import tqdm
import copy
import torch.nn.functional as F


def compress_one_data_part(args, img_mdoel, txt_mdoel, fusion_model, dataloader):
    BI = list()
    BT = list()
    L = list()
    for img_path, img, txt, label in tqdm(dataloader, disable=args.tqdm_disable):
        
        if args.freeze_img_backbone:
            img = img.permute(1, 0, 2)
        img = img.cuda()
        txt = txt.cuda()
        labe = label.cuda()
        # label = F.one_hot(label, num_classes=10).float()
        
        # H_I, cls_out = img_mdoel(img)
        H_I = img_mdoel(img).mean(0)
        H_T = txt_mdoel(txt).mean(0)

        B_I = H_I.sign()
        B_T = H_T.sign()

        BI.extend(B_I)
        BT.extend(B_T)
        L.extend(labe)

        functional.reset_net(img_mdoel)
        functional.reset_net(txt_mdoel)

    BI = torch.stack(BI)
    BT = torch.stack(BT)
    L = torch.stack(L)

    return BI, BT, L

def compress(args, img_mdoel, txt_mdoel, fusion_model, te_loader, db_loader):
    # test
    te_BI, te_BT, te_L = compress_one_data_part(args, img_mdoel, txt_mdoel, fusion_model, te_loader)
    # te_BI, te_L = compress_one_data_part(args, te_loader, img_mdoel)
    
    # database
    if db_loader is te_loader:
        db_BI, db_BT, db_L = te_BI.clone(), te_BT.clone(), te_L.clone()
        # db_BI, db_L = te_BI.clone(), te_L.clone()
    else:
        db_BI, db_BT, db_L = compress_one_data_part(args, img_mdoel, txt_mdoel, fusion_model, db_loader)
        # db_BI, db_L = compress_one_data_part(args, db_loader, img_mdoel)
 
    return te_BI, te_BT, te_L, db_BI, db_BT, db_L
    # return te_BI, te_L, db_BI, db_L


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    map = 0.
    if topk is None:
        topk = re_L.shape[0]

    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        _, ind = torch.sort(hamm, stable=True)  # 默认稳定排序
        ind.squeeze_()
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue

        count = torch.arange(1, int(tsum) + 1).cuda(non_blocking=True).type(torch.float32)
        tindex = torch.nonzero(tgnd).squeeze().type(torch.float32) + 1.0
        map = map + torch.mean(count / tindex)
    
    map = map / num_query
    return map


def validate(args, img_mdoel, txt_model, fusion_model, te_loader, db_loader, topk=500):
    img_mdoel.eval()
    txt_model.eval()
    fusion_model.eval()
    
    with torch.no_grad():
        te_BI, te_BT, te_L, db_BI, db_BT, db_L = compress(args, img_mdoel, txt_model, fusion_model, te_loader, db_loader)

        # te_BI, te_BT, db_BI, db_BT = torch.randn_like(te_BI) - 0.5, torch.randn_like(te_BT) - 0.5, torch.randn_like(db_BI) - 0.5, torch.randn_like(db_BT) - 0.5

        MAP_I2T = calculate_top_map(te_BI, db_BT, te_L, db_L, topk=topk)
        MAP_T2I = calculate_top_map(te_BT, db_BI, te_L, db_L, topk=topk)

        MAP_I2I = calculate_top_map(te_BI, db_BI, te_L, db_L, topk=topk)
        MAP_T2T = calculate_top_map(te_BT, db_BT, te_L, db_L, topk=topk)

    return MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, (te_BI, te_BT, te_L, db_BI, db_BT, db_L)
    # return MAP_I2I

if __name__ == '__main__':
    import scipy.io as scio
    import os
    
    # d = '/data/zhangzhen/logs/torch_resnet/nuswide/DCHMT'
    d = '/data/zhangzhen/logs/my_pretrain_resnet/nuswide/DCHMT'
    mat_files = []
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith('.mat'):
                mat_path = os.path.join(root, file)
                print(mat_path)
                a = scio.loadmat(mat_path)
                te_BI = torch.from_numpy(a['te_BI']).cuda()
                te_BT = torch.from_numpy(a['te_BT']).cuda()
                te_L = torch.from_numpy(a['te_L']).cuda()
                db_BI = torch.from_numpy(a['db_BI']).cuda()
                db_BT = torch.from_numpy(a['db_BT']).cuda()
                db_L = torch.from_numpy(a['db_L']).cuda()

                test_num = te_L.shape[0]
                te_BI = te_BI[:test_num]
                te_BT = te_BT[:test_num] 

                I2T = calculate_top_map(te_BI, db_BT, te_L, db_L, 500)
                T2I = calculate_top_map(te_BT, db_BI, te_L, db_L, 500)
                print(f'I2T: {I2T:.3f}, T2I: {T2I:.3f}')
