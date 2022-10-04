import os
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm, trange

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='/home/hi/ZXH/BiSeNet3-29/configs/bisenetv1_city.py',)
parse.add_argument('--weight-path', type=str, default='/home/hi/ZXH/BiSeNet3-29/res/bisenet1024/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='/home/hi/data/cityscapes/leftImg8bit/val/munster/munster_000020_000019_leftImg8bit.png',)
parse.add_argument('--colors_path', type=str, default='/home/hi/ZXH/BiSeNet3-29/datasets/cityscapes/cityscapes_colors1.txt',)
parse.add_argument('--names_path', type=str, default='/home/hi/ZXH/BiSeNet3-29/datasets/cityscapes/cityscapes_names.txt',)

args = parse.parse_args()
cfg = set_cfg_from_file(args.config)



#palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
#colors_path = 'home/hi/ZXH/BiSeNet3-2/datasets/cityscapes/cityscapes_colors.txt'
colors = np.loadtxt(args.colors_path).astype('uint8')
names = [line.rstrip('\n') for line in open(args.names_path)]


# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
#im = cv2.imread(args.img_path)   # bgr
#im = im[:, :, ::-1]
im = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#im = cv2.imread(args.img_path)[:, :, ::-1]
#im = np.asarray(Image.open(args.img_path), dtype=np.int32)
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# inference
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
pred = colors[out]
cv2.imwrite('./P/munster_000020_000019_leftImg8bit.png', pred)
