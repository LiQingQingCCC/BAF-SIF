import os
import argparse

from tqdm import tqdm
import pandas as pd
import joblib
import glob

from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.net1 import FusionAdversarialModel

from losses import ssim_ir, ssim_vi,RMI_ir,RMI_vi
import numpy as np
import nibabel as nib
import torch.nn as nn
from tqdm import tqdm
import random

torch.cuda.set_device(0)

device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()
print(f"CUDA is available: {torch.cuda.is_available()}")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model name', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('--weight', default=[1, 1,1,2.5], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)  # adam优化器衰减率参数梯度均值和梯度方差
    parser.add_argument('--eps', default=1e-8, type=float)  # 用于数值稳定，防止在计算过程中出现除以零或数值爆炸的问题
    parser.add_argument('--alpha', default=300, type=int,
                        help='number of new channel increases per depth (default: 300)') # 未体现
    args = parser.parse_args()

    return args

# 数据预处理
def process_file_pair(args):
    diffusion_name, folder_path = args
    num = diffusion_name.split('regis')[0]
    diff_path = os.path.join(folder_path, 'diffusion', diffusion_name)
    vessel_path = os.path.join(folder_path, 'vessel', f'{num}restruct.nii.gz')

    if not os.path.exists(diff_path) or not os.path.exists(vessel_path):
        return None, None

    diffs_data = nib.load(diff_path).get_fdata().astype(np.float32)
    diffs_data = np.clip(0,np.percentile(diffs_data[diffs_data>0],80))  
    vessels_data = nib.load(vessel_path).get_fdata().astype(np.float32)

    max_val = diffs_data.max()
    if max_val > 0:
        diffs_data /= max_val

    return diffs_data, vessels_data

# class GetDataset(Dataset):
#     def __init__(self, folder_path,mode='train'):
#         diffs_ls=[]
#         vessels_ls=[]
#         eps=1e-20
#         ls_names=os.listdir(os.path.join(folder_path,'diffusion'))
#         self.mode=mode
#         # if mode=='train':
#         #     iter_names=ls_names[:20]  # 前20个样本用于训练
#         # elif mode=='val':
#         #     iter_names=ls_names[20:25]  # 第20-30个样本用于验证
#         # else:
#         #     iter_names=ls_names[25:30]   # [:30]前30个样本用于测试
        
        
#         if mode=='train':
#             iter_names=ls_names[:9]  # 前20个样本用于训练
#         elif mode=='val':
#             iter_names=ls_names[10:11]  # 第20-30个样本用于验证
#         else:
#             iter_names=ls_names[12:13]   # [:30]前30个样本用于测试
        
#         self.iter_names=iter_names
            
#         for name in tqdm(iter_names):
#             num=name.split('regis')[0]
#             diffs=nib.load(os.path.join(folder_path,'diffusion',name)).get_fdata().astype(float) #H,W,N
#             vessels=nib.load(os.path.join(folder_path,'vessel',f'{num}restruct.nii.gz')).get_fdata().astype(float)
            
#             diffs = np.clip(diffs,0,np.percentile(diffs[diffs>0],95))            
#             diffs/=diffs.max()
#             vessels/=vessels.max()
#             diffs_ls.append(diffs)
#             vessels_ls.append(vessels)
#         self.diffs=np.concatenate(diffs_ls,axis=-1)
#         self.vessels=np.concatenate(vessels_ls,axis=-1)
#         self.diffs_ls=diffs_ls
#         self.vessels_ls=vessels_ls
#         # self.transforms_diff=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(384,384),interpolation=transforms.InterpolationMode.BILINEAR)])
#         # self.transforms_vessel=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(384,384),interpolation=transforms.InterpolationMode.NEAREST)])
#         self.transforms_diff=transforms.Compose([transforms.ToTensor()])
#         self.transforms_vessel=transforms.Compose([transforms.ToTensor()])


   

#     def __getitem__(self, index):
#         if self.mode=='test':
#             diff=self.transforms_diff(self.diffs_ls[index])
#             vessel=self.transforms_vessel(self.vessels_ls[index])
#             return {'diff':diff.unsqueeze(1),'vessel':vessel.unsqueeze(1),'name':self.iter_names[index]}
        
#         else:
#             diff=self.transforms_diff(self.diffs[...,index])
#             vessel=self.transforms_vessel(self.vessels[...,index])
#             return diff,vessel

#     def __len__(self):
#         if self.mode=='test':
#             return len(self.diffs_ls)
#         else:
#             return self.diffs.shape[-1]
    
# 新数据训练使用
# class GetDataset(Dataset):
#     def __init__(self, folder_path, mode='train'):
#         diffs_ls = []
#         vessels_ls = []
#         eps = 1e-20
        
#         diffusion_dir = os.path.join(folder_path, 'diffusion')
#         ls_names = os.listdir(diffusion_dir)
        
#         self.mode = mode
        
#         # 根据模式选择样本
#         if mode == 'train':
#             iter_names = ls_names[:3]
#         elif mode == 'val':
#             iter_names = ls_names[4:5]
#         else:  # test模式
#             iter_names = ls_names[-1]
            
#         self.iter_names = iter_names
        
#         for name in tqdm(iter_names):
#             # 精确提取编号部分（修正点）
#             if 'regis' in name:
#                 # 1. 先去掉 `.nii.gz` 后缀
#                 name_without_ext = name.replace('.nii.gz', '')
#                 # 2. 再提取 `xxxx_xx` 部分
#                 num = name_without_ext.split('regis')[0]
#                 print('num:', num)  # 应该是 "1715_13" 而不是 "1715_13regis.nii.gz"
#             else:
#                 print(f"Skipping file {name} as it doesn't contain 'regis'")
#                 continue
            
#             # 加载 diffusion 数据
#             diffs_path = os.path.join(diffusion_dir, name)
#             diffs = nib.load(diffs_path).get_fdata().astype(float)
            
#             # 构造对应的 vessel 文件名（修正点）
#             vessel_dir = os.path.join(folder_path, 'vessel')
#             vessel_name = f"{num}restruct.nii.gz"  # 假设血管文件命名格式是 "xxxx_xx_restruct.nii.gz"
#             vessels_path = os.path.join(vessel_dir, vessel_name)
            
#             # 检查文件是否存在
#             if not os.path.exists(vessels_path):
#                 print(f"Skipping {name} - 对应的vessel文件 {vessel_name} 不存在")
#                 continue
                
#             # 加载 vessel 数据
#             vessels = nib.load(vessels_path).get_fdata().astype(float)
            
#             # 数据处理
#             diffs = np.clip(diffs, 0, np.percentile(diffs[diffs > 0], 95))
#             diffs /= (diffs.max() + eps)
#             vessels /= (vessels.max() + eps)
            
#             diffs_ls.append(diffs)
#             vessels_ls.append(vessels)
        
#         # 验证数据加载
#         if not diffs_ls or not vessels_ls:
#             raise ValueError(f"No data loaded for mode '{mode}'. Check dataset path or naming format.")
        
#         self.diffs = np.concatenate(diffs_ls, axis=-1)
#         self.vessels = np.concatenate(vessels_ls, axis=-1)
#         self.diffs_ls = diffs_ls
#         self.vessels_ls = vessels_ls
        
#         # 数据变换
#         self.transforms_diff = transforms.Compose([transforms.ToTensor()])
#         self.transforms_vessel = transforms.Compose([transforms.ToTensor()])


#     def __getitem__(self, index):
#         if self.mode == 'test':
#             diff = self.transforms_diff(self.diffs_ls[index])
#             vessel = self.transforms_vessel(self.vessels_ls[index])
#             return {'diff': diff.unsqueeze(1), 'vessel': vessel.unsqueeze(1), 'name': self.iter_names[index]}
#         else:
#             diff = self.transforms_diff(self.diffs[..., index])
#             vessel = self.transforms_vessel(self.vessels[..., index])
#             return diff, vessel

#     def __len__(self):
#         if self.mode == 'test':
#             return len(self.diffs_ls)
#         else:
#             return self.diffs.shape[-1]
            
# 新数据验证使用
class GetDataset(Dataset):
    def __init__(self, folder_path, mode='train'):
        self.diffs_ls = []
        self.vessels_ls = []
        self.eps = 1e-20
        
        diffusion_dir = os.path.join(folder_path, 'diffusion')
        vessel_dir = os.path.join(folder_path, 'vessel')
        
        # 获取所有符合条件的文件
        diff_files = [f for f in os.listdir(diffusion_dir) if 'regis' in f and f.endswith('.nii.gz')]
        vessel_files = {f.replace('regis', 'restruct').replace('.nii.gz', '_restruct.nii.gz') 
                        # 或者更简单的：直接匹配编号部分，如：
                        # 实际上，根据文件名示例，vessel 文件名应为 "xxxx_xxrestruct.nii.gz"
                        # 所以这里可以重新构造 vessel 文件名
                        : None for f in diff_files}  # 临时占位，实际不需要
        # 更正：直接遍历 diff_files，并构造对应的 vessel 文件名
        vessel_file_dict = {}
        for f in diff_files:
            num = f.split('regis')[0]
            vessel_name = f"{num}restruct.nii.gz"  # 构造 vessel 文件名
            vessel_file_dict[f] = vessel_name

        # 根据模式选择文件
        if mode == 'train':
            selected_diff = diff_files[:3]
        elif mode == 'val':
            selected_diff = diff_files[4]
        elif mode == 'test':
            # selected_diff = diff_files[5:6]  # 取最后一个文件
            selected_diff = diff_files[:]  # 取最后一个文件
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        # 存储选中的文件名
        self.iter_names = selected_diff
        
        # 加载数据
        for diff_name in tqdm(selected_diff):
            # 构造对应的 vessel 文件名
            vessel_name = vessel_file_dict[diff_name]
            vessel_path = os.path.join(vessel_dir, vessel_name)
            
            # 检查文件是否存在
            if not os.path.exists(vessel_path):
                print(f"Warning: Vessel file {vessel_path} not found, skipping {diff_name}")
                continue
                
            # 加载 diffusion 数据
            diff_path = os.path.join(diffusion_dir, diff_name)
            try:
                diffs = nib.load(diff_path).get_fdata().astype(np.float32)
                vessels = nib.load(vessel_path).get_fdata().astype(np.float32)
                
                # 数据预处理
                diffs = np.clip(diffs, 0, np.percentile(diffs[diffs > 0], 95))
                diffs /= (diffs.max() + self.eps)
                vessels /= (vessels.max() + self.eps)
                
                self.diffs_ls.append(diffs)
                self.vessels_ls.append(vessels)
            except Exception as e:
                print(f"Error loading {diff_name} or {vessel_name}: {str(e)}")
        
        # 验证数据加载
        if not self.diffs_ls or not self.vessels_ls:
            raise ValueError(f"No data loaded for mode '{mode}'. Check dataset path or naming format.")
        
        # 数据变换
        self.transforms_diff = transforms.Compose([transforms.ToTensor()])
        self.transforms_vessel = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        diff = self.transforms_diff(self.diffs_ls[index])
        vessel = self.transforms_vessel(self.vessels_ls[index])
        return {'diff': diff.unsqueeze(1), 'vessel': vessel.unsqueeze(1), 'name': self.iter_names[index]}

    def __len__(self):
        return len(self.diffs_ls)
            
# 计算平均值
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion_rec, criterion_gan,optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_gan = AverageMeter()

    model.train()
    eps=1e-8

    for i, (diff,vessel)  in tqdm(enumerate(train_loader), total=len(train_loader)):
        ez_fuse=(diff*4+vessel)/5  # 目标融合图像  ez_fuse：融合扩散图像和血管图像（加权平均）
        diff=diff.type(torch.float32).cuda()
        vessel=vessel.type(torch.float32).cuda()
        ez_fuse=ez_fuse.type(torch.float32).cuda()
        ref_data=torch.full_like(diff,0,device='cpu')  # 创建全0张量用于重建损失计算
        
        # 生成器生成重建图像和特征
        pRE_diff, pRE_vessel,pRE_fusion, feat_diff,feat_vessel,feat_fusion = model.generator(diff, vessel)  
        # 判别器对生成的特征进行分类
        disc_output_diff = model.discriminator(feat_diff)
        disc_output_vessel = model.discriminator(feat_vessel)
        disc_output_fusion = model.discriminator(feat_fusion)
        
        # 计算重建损失
        ezfuse_cpu=ez_fuse.detach().cpu()
        ref_data_cuda=ref_data.cuda()
        
        if (ezfuse_cpu>eps).sum()!=0:
            # 计算参考损失（基于 ref_data）
            loss_rec_diff_ref=criterion_rec(diff[ezfuse_cpu>eps].detach().cpu(),ref_data[ezfuse_cpu>eps]+0.5)
            loss_rec_vessel_ref=criterion_rec(vessel[ezfuse_cpu>eps].detach().cpu(),ref_data[ezfuse_cpu>eps]+0.5)
            loss_rec_fusion_ref=criterion_rec(ez_fuse[ezfuse_cpu>eps].detach().cpu(),ref_data[ezfuse_cpu>eps]+0.5)
            # 计算重建损失（基于生成器输出）
            loss_rec_diff=criterion_rec(diff[ez_fuse>eps],pRE_diff[ez_fuse>eps])
            loss_rec_vessel=criterion_rec(vessel[ez_fuse>eps],pRE_vessel[ez_fuse>eps])
            loss_rec_fusion=criterion_rec(ez_fuse[ez_fuse>eps],pRE_fusion[ez_fuse>eps])
            # 添加额外损失项（基于 ref_data）
            loss_rec_diff=loss_rec_diff+criterion_rec(ref_data_cuda[ez_fuse<=eps],pRE_diff[ez_fuse<=eps])*min(epoch,5)
            loss_rec_vessel=loss_rec_vessel+criterion_rec(ref_data_cuda[ez_fuse<=eps],pRE_vessel[ez_fuse<=eps])*min(epoch,5)
            loss_rec_fusion=loss_rec_fusion+criterion_rec(ref_data_cuda[ez_fuse<=eps],pRE_fusion[ez_fuse<=eps])*min(epoch,15)
        else:
            # 如果 ez_fuse 全为零，使用默认损失
            loss_rec_diff_ref=torch.tensor(0.25)
            loss_rec_vessel_ref=torch.tensor(0.25)
            loss_rec_fusion_ref=torch.tensor(0.25)

            loss_rec_diff=criterion_rec(diff,pRE_diff)
            loss_rec_vessel=criterion_rec(vessel,pRE_vessel)
            loss_rec_fusion=criterion_rec(ez_fuse,pRE_fusion)

            loss_rec_diff=loss_rec_diff
            loss_rec_vessel=loss_rec_vessel
            loss_rec_fusion=loss_rec_fusion
            

        # loss_rec_diff=criterion_rec(diff[ez_fuse>eps],pRE_diff[ez_fuse>eps])
        # loss_rec_vessel=criterion_rec(vessel[ez_fuse>eps],pRE_vessel[ez_fuse>eps])
        # loss_rec_fusion=criterion_rec(ez_fuse[ez_fuse>eps],pRE_fusion[ez_fuse>eps])
        
        # 计算总的重建损失
        loss_rec=loss_rec_diff+loss_rec_vessel+loss_rec_fusion*3
        # 计算对抗损失
        loss_gan=criterion_gan(disc_output_diff,torch.full_like(disc_output_diff,1,device='cuda').float()) \
            +criterion_gan(disc_output_vessel,torch.full_like(disc_output_vessel,0,device='cuda').float()) \
            +criterion_gan(disc_output_fusion,torch.full_like(disc_output_fusion,0.5,device='cuda').float())
        
        
        if epoch<70:
            weight_gan=0
        else:
            weight_gan=min((epoch-70)*0.005,0.05)
        loss=loss_rec+loss_gan*weight_gan


        losses.update(loss.item(), diff.size(0))
        losses_rec.update(loss_rec.item(), diff.size(0))
        losses_gan.update(loss_gan.item(), diff.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('losses_rec', losses_rec.avg),
        ('losses_gan', losses_gan.avg),

    ])

    return log

@torch.no_grad()
def val(args, val_loader, model, criterion_rec, epoch, scheduler=None):
    losses_rec_ref = AverageMeter()
    losses_rec = AverageMeter()

    model.eval()
    eps=1e-10

    for i, (diff,vessel)  in tqdm(enumerate(val_loader), total=len(val_loader)):
        ez_fuse=(diff*4+vessel)/10

        diff=diff.type(torch.float32).cuda()
        vessel=vessel.type(torch.float32).cuda()
        ez_fuse=ez_fuse.type(torch.float32).cuda()
        ref_data=torch.full_like(diff,0,device='cpu')
        pRE_diff, pRE_vessel,pRE_fusion, feat_diff,feat_vessel,feat_fusion = model.generator(diff, vessel)
        ezfuse_cpu=ez_fuse.detach().cpu()
        ref_data_cuda=ref_data.cuda()
        if (ezfuse_cpu>eps).sum()!=0:
            loss_rec_fusion_ref=criterion_rec(ez_fuse[ezfuse_cpu>eps].detach().cpu(),ref_data[ezfuse_cpu>eps]+0.5)
            loss_rec_fusion=criterion_rec(ez_fuse[ez_fuse>eps],pRE_fusion[ez_fuse>eps])

            loss_rec=loss_rec_fusion
        else:
            loss_rec_fusion_ref=torch.tensor(0.25)
            loss_rec_fusion=criterion_rec(ez_fuse,pRE_fusion)

            loss_rec=loss_rec_fusion
            

        losses_rec.update(loss_rec.item(), ez_fuse.size(0))
        losses_rec_ref.update(loss_rec_fusion_ref.item(),ez_fuse.size(0))
        print(loss_rec.item(),loss_rec_fusion_ref.item())

    log = OrderedDict([
        ('losses_rec', losses_rec.avg),
        ('losses_rec_ref', losses_rec_ref.avg),
    ])

    return log

@torch.no_grad()
def inference(model,diff,vessel,name,save_path='./fusion_results'):
    model.eval()
    chunk_size=64
    D = diff.shape[0]
    num_chunks = (D + chunk_size - 1) // chunk_size  # 计算需要分割的块数
    fusion_chunks = []
    ref_chunks=[]
    ez_fuse=((diff*9+vessel)/10).float().cuda()
    l1loss=nn.L1Loss()
    eps=1e-8

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, D)
        ez_fuse_chunk=ez_fuse[start:end]
        diff_chunk = diff[start:end]
        vessel_chunk = vessel[start:end]
        with torch.no_grad():
            _, _, pRE_fusion_chunk, _, _, _ = model.generator(diff_chunk, vessel_chunk)
        pRE_fusion_chunk = torch.clamp(pRE_fusion_chunk, 0, 1)  # 归一化到0-1 # (chunk_D)*H*W
        loss_rec_fusion=l1loss(ez_fuse_chunk[ez_fuse_chunk>eps],pRE_fusion_chunk[ez_fuse_chunk>eps])
        print('loss:',loss_rec_fusion)
        fusion_chunks.append(pRE_fusion_chunk)
        ref_chunks.append(ez_fuse_chunk)
    pRE_fusion = torch.cat(fusion_chunks, dim=0).squeeze()
    ref=torch.cat(ref_chunks, dim=0).squeeze()

    nonzero_indices = (ref != 0).nonzero(as_tuple=True)
    min_d, max_d = nonzero_indices[0].min(), nonzero_indices[0].max()
    min_h, max_h = nonzero_indices[1].min(), nonzero_indices[1].max()
    min_w, max_w = nonzero_indices[2].min(), nonzero_indices[2].max()

    masked_pRE_fusion = torch.zeros_like(pRE_fusion)

    masked_pRE_fusion[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = pRE_fusion[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
    pRE_fusion = masked_pRE_fusion
    # pRE_fusion[ref<eps]=0


    folder_path='./dataset/train/diffusion'
    nii_ori=nib.load(os.path.join(folder_path,name))
    ori_img=nii_ori.get_fdata()
    ori_img_min,ori_img_max=0,np.percentile(ori_img[ori_img>0],95)
    pRE_fusion=pRE_fusion.permute(1,2,0).detach().cpu().numpy() # H*W*D
    
    pRE_fusion=pRE_fusion*ori_img_max

    ref=ref.permute(1,2,0).detach().cpu().numpy()
    ref=ref*ori_img_max
    print(pRE_fusion.shape,pRE_fusion.max())
    # save nii.gz to save_path, header is the same as nii_ori
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, f'fusion_{name}')
    new_nii = nib.Nifti1Image(pRE_fusion, nii_ori.affine, nii_ori.header)
    nib.save(new_nii, save_name)
    save_name = os.path.join(save_path, f'ref_fusion_{name}')
    new_nii = nib.Nifti1Image(ref, nii_ori.affine, nii_ori.header)
    nib.save(new_nii, save_name)
    print(f"Saved fusion result to: {save_name}")
    


@torch.no_grad()
def evaluate():
    training_dir = "/home/230320040445/MI-Fusion/Ours/dataset/train"
    test_loader=DataLoader(GetDataset(training_dir,mode='test'),batch_size=1,shuffle=False)
    
    
    IMG_SIZE = 512
    # PATCH_SIZE = 16
    IN_CHANS = 1 
    ENCODER_NAME='convnext_tiny'

    model = FusionAdversarialModel(
        encoder_name=ENCODER_NAME,
        img_size=IMG_SIZE,
        in_chans=IN_CHANS,
        pretrained=False   # False 无联网状态 
    )

    model=model.cuda().eval()
    with torch.no_grad():
        x=torch.randn(1,1,512,512).cuda().float()
        model(x,x)

    pth_path='/home/230320040445/MI-Fusion/Ours/models/model name/model_best_model.pth'
    # model.load_state_dict(torch.load(pth_path),'cuda')
     # 修改这里：添加map_location参数  GPU
    model.load_state_dict(torch.load(pth_path, map_location='cuda:0'))  
    model=model.cuda().eval()
    for loader in tqdm(test_loader):
        loader_dict=loader
        diff=loader_dict['diff'].cuda().float()[0]
        vessel=loader_dict['vessel'].cuda().float()[0]
        name=loader_dict['name'][0]
        print(diff.max(),diff.min(),vessel.max(),vessel.min(),name)
        inference(model,diff,vessel,name)




def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    training_dir = "/home/230320040445/MI-Fusion/Ours/dataset/train"
    train_loader=DataLoader(GetDataset(training_dir,mode='train'),batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(GetDataset(training_dir,mode='val'),batch_size=args.batch_size*4,shuffle=False)
    
    IMG_SIZE = 512
    # PATCH_SIZE = 16
    IN_CHANS = 1 
    # ENCODER_NAME = 'vit_base_patch16_384' 
    ENCODER_NAME='convnext_tiny'
    # model = FusionAdversarialModel(
    #     encoder_name=ENCODER_NAME,
    #     img_size=IMG_SIZE,
    #     patch_size=PATCH_SIZE,
    #     in_chans=IN_CHANS,
    #     embed_dim=768,
    #     pretrained=True 
    # )
    model = FusionAdversarialModel(
        encoder_name=ENCODER_NAME,
        img_size=IMG_SIZE,
        in_chans=IN_CHANS,
        pretrained=False # 使用预训练权重   False 无联网状态
    )

    model=model.cuda()
    criterion_rec=nn.L1Loss()
    criterion_gan=nn.BCEWithLogitsLoss()
    
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'loss',
                                'losses_rec',
                                'losses_gan',
                                ])
    log_val = pd.DataFrame(index=[],
                       columns=['epoch',
                                'losses_rec',
                                'losses_rec_ref',
                                ])

    loss_min=10000

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))

        train_log = train(args, train_loader, model, criterion_rec,criterion_gan, optimizer, epoch)     # 训练集
        val_log = val(args, val_loader, model, criterion_rec, epoch)
        print('loss: %.4f - losses_rec: %.4f - losses_gan: %.4f  '
              % (train_log['loss'],
                 train_log['losses_rec'],
                 train_log['losses_gan'],
                 ))

        tmp = pd.Series([
            epoch + 1,
            train_log['loss'],
            train_log['losses_rec'],
            train_log['losses_gan'],

        ], index=['epoch', 'loss', 'losses_rec', 'losses_gan'])

        # log = log.append(tmp, ignore_index=True)   # 版本1.5
        # 修改为（Pandas 2.0+）
        log = pd.concat([log, tmp.to_frame().T], ignore_index=True)

        log.to_csv('models/%s/log.csv' %args.name, index=False)
        
        tmp = pd.Series([
            epoch + 1,
            val_log['losses_rec'],
            val_log['losses_rec_ref'],

        ], index=['epoch', 'losses_rec', 'losses_rec_ref'])
        
        # log_val = log_val.append(tmp, ignore_index=True)  # 版本1.5
        # 修改为（Pandas 2.0+）
        log_val = pd.concat([log_val, tmp.to_frame().T], ignore_index=True)

        log_val.to_csv('models/%s/log_val.csv' %args.name, index=False)
        
        loss_rec_=val_log['losses_rec']
        if loss_rec_<loss_min:
            loss_min=loss_rec_
            torch.save(model.state_dict(), f'models/{args.name}/model_best_model.pth')


if __name__ == '__main__':
    seed_value = 42
    set_seed(seed_value)
    # main()
    evaluate()