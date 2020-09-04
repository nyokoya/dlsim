# =============================================================================
# Import libs
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import tqdm
import cv2
import argparse
import random
from tifffile import imread, imwrite
from utils.geotiffs import Geotiffs
import utils.transforms as transforms
import models.networks as nw

# =============================================================================
# Parameters
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--data',default='NK2017', help='NK2017 or WJ2018')
parser.add_argument('--type', default='wl', help='wl (water level) or td (topographic deformation)')
parser.add_argument('--train_test', default='train')
parser.add_argument('--data_dir', default='./data/', help='data directory')
parser.add_argument('--architecture', default='attunet', help='attunet or linknet')
parser.add_argument('--loss', default='smoothl1', help='mse or l1 or smoothl1')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--data_split', default='all', help='scenario-wise or all')
args = parser.parse_args()

data_dir = os.path.join(args.data_dir,args.data) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_ch = 1
filename = args.type+'_'+args.architecture+'_'+str(args.batch_size)+'_'+str(args.epochs)

print('===================================')
print('data            : ', args.data)
print('mode            : ', args.train_test)
print('target variable : ', args.type)
print('loss            : ', args.loss)
print('architecture    : ', args.architecture)
print('epochs          : ', args.epochs)
print('===================================')

if args.train_test == 'test':
    args.epochs = 0

# path to save models and images
model_dir = os.path.join(data_dir,'model')
output_dir = os.path.join(data_dir,'output')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# =============================================================================
# Transformations
# =============================================================================

noise_param = 0.995
n_holes = 1
length = 128
p = 0.5

if args.type == 'wl':
    in_ch = 2
    train_transforms = torchvision.transforms.Compose([
            transforms.Cutout(n_holes, length, p, ['wl','td','bc']),
            transforms.AdditiveNoise(noise_param,1,['bc']),
            transforms.AdditiveNoise(noise_param,2,['bc']),
            transforms.Stack(['bc','sl'],['wl']),
            transforms.ToTensor(),
    ])
    test_transforms = torchvision.transforms.Compose([
            transforms.AdditiveNoise(noise_param,1,['bc']),
            transforms.AdditiveNoise(noise_param,2,['bc']),
            transforms.Stack(['bc','sl'],['wl']),
            transforms.ToTensor(),
    ])
    test_real_transforms = torchvision.transforms.Compose([
            transforms.Stack(['bc','sl'],['wl']),
            transforms.ToTensor(),
    ])
elif args.type == 'td':
    in_ch = 2
    train_transforms = torchvision.transforms.Compose([
            transforms.Cutout(n_holes, length, p, ['wl','td','bc']),
            transforms.AdditiveNoise(noise_param,1,['bc']),
            transforms.AdditiveNoise(noise_param,2,['bc']),
            transforms.Stack(['bc','sl'],['td']),
            transforms.ToTensor(),
    ])
    test_transforms = torchvision.transforms.Compose([
            transforms.AdditiveNoise(noise_param,1,['bc']),
            transforms.AdditiveNoise(noise_param,2,['bc']),
            transforms.Stack(['bc','sl'],['td']),
            transforms.ToTensor(),
    ])
    test_real_transforms = torchvision.transforms.Compose([
            transforms.Stack(['bc','sl'],['td']),
            transforms.ToTensor(),
    ])
        

# =============================================================================
# Create model and train
# =============================================================================

if args.architecture == 'attunet':
    model = nw.AttU_Net(img_ch=in_ch,output_ch=out_ch)
    model.apply(nw.weights_init_xavier)
elif args.architecture == 'linknet':
    model = nw.Link_Net(img_ch=in_ch,output_ch=out_ch)
    model.apply(nw.weights_init_xavier)
    
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model =  nn.DataParallel(model)
else:
    print('Only 1 GPU available')
    
model.to(device)

# Print model parameter size
nw.print_network(model)

def train_model(model, criterion, optimizer, loader, n_epochs=25):
    pbar = tqdm.tqdm(total=n_epochs)
    for epoch in range(1, n_epochs + 1):

        running_loss = 0.0
        for x, y in loader:

            img_in = x.to(device).float()
            img_out = y.to(device).float()

            optimizer.zero_grad()

            output = model.forward(img_in)
            loss = criterion(output, img_out)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img_in.size(0)

        running_loss /= len(loader.dataset)

        info_str = "epoch {:3d}, t_loss:{:7.4f}".format(epoch, running_loss)
        pbar.update(1)
        pbar.set_description(info_str)
        pbar.write(info_str)

    return model

if args.data_split == 'scenario-wise':
    train_data = {
        'wl': 'mosaic/wl_scenario-wise_train.tif',
        'td': 'mosaic/td_scenario-wise_train.tif',
        'bc': 'mosaic/bc_scenario-wise_train.tif',
        'sl': 'mosaic/sl_scenario-wise_train.tif',
    }
    test_data = {
        'wl': 'mosaic/wl_scenario-wise_test.tif',
        'td': 'mosaic/td_scenario-wise_test.tif',
        'bc': 'mosaic/bc_scenario-wise_test.tif',
        'sl': 'mosaic/sl_scenario-wise_test.tif',
    }
else:
    train_data = {
        'wl': 'mosaic/wl.tif',
        'td': 'mosaic/td.tif',
        'bc': 'mosaic/bc.tif',
        'sl': 'mosaic/sl.tif',
    }
    test_data = {
        'wl': 'mosaic/wl_scenario-wise_test.tif',
        'td': 'mosaic/td_scenario-wise_test.tif',
        'bc': 'mosaic/bc_scenario-wise_test.tif',
        'sl': 'mosaic/sl_scenario-wise_test.tif',
    }
    
if args.loss == 'smoothl1':
    criterion = torch.nn.SmoothL1Loss()
if args.loss == 'l1':
    criterion = torch.nn.L1Loss()
if args.loss == 'mse':
    criterion = torch.nn.MSELoss()

if args.epochs > 0:
    with Geotiffs(data_dir, args.img_size, args.img_size, transform=train_transforms, **train_data) as train_dataset:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        train_model(model, criterion, optimizer, train_loader, args.epochs)

        if torch.cuda.device_count() > 1:
            # if multiple GPUs is used
            torch.save(model.module.state_dict(), os.path.join(model_dir,filename+'.pth'))
        else:
            torch.save(model, os.path.join(model_dir,filename+'.pth'))

        print('Training done')

# =============================================================================
# Evaluation metrics
# =============================================================================

def rmse(gt,pr):
    return np.mean((gt.flatten() - pr.flatten())**2)**0.5

def log_histogram_intersection(gt, pr):
        B = 400
        h1, _ = np.histogram(gt.flatten(),bins=B, range=(-20,20))
        h2, _ = np.histogram(pr.flatten(),bins=B, range=(-20,20))
        h1[h1<1] = 1
        h2[h2<1] = 1
        h1 = np.log(h1)
        h2 = np.log(h2)
        sm = 0
        S = 0
        for i in range(B):
            sm += min(h1[i], h2[i])
            S += max(h1[i], h2[i])   
        return sm/S

def iou(pr, gt, eps=1e-7):
    intersection = np.sum(gt * pr)
    union = np.sum(gt) + np.sum(pr) - intersection + eps
    return (intersection + eps) / union

# =============================================================================
# Load pretrained model
# =============================================================================

if torch.cuda.device_count() > 1:
    print('Multiple GPUs')
    # if multiple GPUs is used
    device = torch.device("cuda")
    if args.architecture == 'attunet':
        pretrained_model = nw.AttU_Net(img_ch=in_ch,output_ch=out_ch)
    elif args.architecture == 'linknet':
        pretrained_model = nw.Link_Net(img_ch=in_ch,output_ch=out_ch)
          
    pretrained_model.load_state_dict(torch.load(os.path.join(model_dir,filename+'.pth'), map_location="cuda:0")) 
    pretrained_model.to(device)
else:
    print('Only 1 GPU')
    pretrained_model = torch.load(os.path.join(model_dir,filename+'.pth'))
    
pretrained_model.eval()

# =============================================================================
# Evaluation with synthetic data
# =============================================================================

stride_size = 128

with Geotiffs(data_dir, args.img_size, stride_size, transform=test_transforms, **test_data) as test_dataset:
    gt = imread(os.path.join(data_dir,test_data[args.type]))
    
    H = gt.shape[0]
    W = gt.shape[1]

    R = (H - args.img_size) // stride_size + 1
    C = (W - args.img_size) // stride_size + 1
    
    pre = np.zeros((H,W,out_ch))
    
    mask = np.zeros((H,W,1))
    
    print('num of test samples: ', len(test_dataset))
    print('tile size: ', R, C)
        
    for n in range(len(test_dataset)):

        input, output = test_dataset[n]
        input = input.to(device).unsqueeze(0)
                
        prediction = pretrained_model.predict(input)   
        prediction = prediction.squeeze().cpu().numpy()
        
        r = n // C
        c = n%C 
        
        pre[r*stride_size:(r*stride_size+args.img_size),c*stride_size:(c*stride_size+args.img_size),0] += prediction
        mask[r*stride_size:(r*stride_size+args.img_size),c*stride_size:(c*stride_size+args.img_size),:] += 1
     
    mask[mask==0] = 1
    pre = pre / mask
    
    for b in range(out_ch):
        if out_ch > 1:
            ref = gt[0:(R-1)*stride_size+args.img_size,0:(C-1)*stride_size+args.img_size,b]
        else:
            ref = gt[0:(R-1)*stride_size+args.img_size,0:(C-1)*stride_size+args.img_size]
        tar = pre[0:(R-1)*stride_size+args.img_size,0:(C-1)*stride_size+args.img_size,b]
        
        rmse_score = rmse(ref,tar)
        hi_score = log_histogram_intersection(ref,tar)

        print('Evaluation results with simulation data')
        print('RMSE: ', rmse_score)
        print('LSHI: ', hi_score)
        
        
    if args.data_split != 'all':
        imwrite(os.path.join(output_dir,filename+'_synthetic.tif'),pre)     
    
# =============================================================================
# Evaluation with real data
# =============================================================================

import time 

if args.data == 'NK2017':
    test_real_data = {
        'wl': 'SIM/wl_case0.tif',
        'td': 'SIM/td_case0.tif',
        'bc': 'RS/binary_change_dilate.tif',
        'sl': 'DEM/slope.tif',
    }
elif args.data == 'WJ2018':
    test_real_data = {
        'wl': 'SIM/wl_case101.tif',
        'td': 'SIM/td_case101.tif',
        'bc': 'RS/binary_change_dilate.tif',
        'sl': 'DEM/slope.tif',
    }    
    
stride_size = 8

with Geotiffs(data_dir, args.img_size, stride_size, transform=test_real_transforms, **test_real_data) as test_real_dataset:
    
    bc_gt = imread(os.path.join(data_dir,'GT/gt.tif'))
    
    gt = imread(os.path.join(data_dir,test_real_data[args.type]))
    valid = valid.squeeze()
    
    H = bc_gt.shape[0]
    W = bc_gt.shape[1]
    
    if args.data == 'NK2017':
        valid = imread(os.path.join(data_dir,'GT/mask.tif'))
    elif args.data == 'WJ2018':
        valid = np.ones((H,W))

    R = (H - args.img_size) // stride_size + 1
    C = (W - args.img_size) // stride_size + 1
    
    pre = np.zeros((H,W,out_ch))
    
    mask = np.zeros((H,W,1))
    
    print('num of test samples: ', len(test_real_dataset))
    print('tile size: ', R, C)
    
    t1 = time.time() 
        
    for n in range(len(test_real_dataset)):

        input, output = test_real_dataset[n]
        input = input.to(device).unsqueeze(0)
        
        prediction = pretrained_model.predict(input)   
        prediction = prediction.squeeze().cpu().numpy()
        
        r = n // C
        c = n%C 
        
        pre[r*stride_size:(r*stride_size+args.img_size),c*stride_size:(c*stride_size+args.img_size),0] += prediction
        mask[r*stride_size:(r*stride_size+args.img_size),c*stride_size:(c*stride_size+args.img_size),:] += 1
    
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"test time：{elapsed_time}")

    mask[mask==0] = 1
    pre = pre / mask


    ref = gt[0:(R-1)*stride_size+args.img_size,0:(C-1)*stride_size+args.img_size]
    tar = pre[0:(R-1)*stride_size+args.img_size,0:(C-1)*stride_size+args.img_size,0]

    rmse_score_realdata = rmse(ref[valid>0],tar[valid>0])
    th = np.percentile(np.abs(tar[valid>0]),80)
    iou_score_realdata = iou(np.abs(tar[valid>0])>th,bc_gt[valid>0]>1)
    hi_score_realdata = log_histogram_intersection(ref[valid>0],tar[valid>0])

    print('Evaluation results with real data')
    print('RMSE: ', rmse_score_realdata) 
    print('IoU : ', iou_score_realdata)
    print('LSHI: ', hi_score_realdata)
        
    imwrite(os.path.join(output_dir,filename+'_real.tif'),pre)    