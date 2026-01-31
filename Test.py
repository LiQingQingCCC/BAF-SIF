from PIL import Image
import numpy as np
import os
import torch
import argparse
import time
import imageio

import torchvision.transforms as transforms

from Networks.net import MODEL as net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')


model = net(in_channel=2)

model_path = "models/model_10.pth"
use_gpu = torch.cuda.is_available()


if use_gpu:

    model = model.cuda()
    model.cuda()

    model.load_state_dict(torch.load(model_path))

else:

    state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--diff_data_csv', type=str, default='')
    parser.add_argument('--vessel_data_csv', type=str, default='')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[1, 1,1,2.5], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--alpha', default=300, type=int,
                        help='number of new channel increases per depth (default: 300)')
    args = parser.parse_args()

    return args

def fusion():

    for num in range(1):
        tic = time.time()

        path1 = './source images/SPECT_001.bmp'

        path2 = './source images/MRI_001.bmp'

        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')


        img1_org = img1
        img2_org = img2

        tran = transforms.ToTensor()

        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)
        if use_gpu:
            input_img = input_img.cuda()
        else:
            input_img = input_img

        model.eval()
        out = model(input_img)

        d = np.squeeze(out.detach().cpu().numpy())
        result = (d* 255).astype(np.uint8)
        imageio.imwrite('./fusion result/{}.bmp'.format(num),
                        result)


        toc = time.time()
        print('end  {}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))



if __name__ == '__main__':

    fusion()
