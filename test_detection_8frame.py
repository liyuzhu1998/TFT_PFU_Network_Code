# from __future__ import print_function
from torch.utils import data
import random
import os
from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import scipy.ndimage as ndimage
import getopt
import sys
from configobj import ConfigObj
from tqdm import tqdm
import os
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torchvision.utils as vutils
import torch.nn.functional as F
import pickle
# from Dataset import get_data_set
import network_v3 as network
import time
import shutil

import h5py as h5
from pathlib import Path
from tqdm import tqdm
import scipy.io


class DataFolder(data.Dataset):


    def __init__(self,in_channels, in_frames, inputstack, time_frame, stepsize=20, cropsize=100, input_transform=None, well_num=1):
        super(DataFolder,self).__init__()

        # inputstack: 2 * time * height * width

        self.in_channels = in_channels
        self.time_frame = time_frame

        self.stepsize = stepsize
        self.cropsize = cropsize

        self.in_frames = in_frames

        time_start = time.time()
        self.input_transform = input_transform

        if in_frames == 1:
            inputstack = inputstack[:, time_frame, :, :]
            self.phase_stack = np.expand_dims(inputstack, axis=1)
        elif in_frames == 2:
            inputstack = inputstack[:, [time_frame-3,time_frame],:, :]
            self.phase_stack = inputstack
        elif in_frames == 8:
            inputstack = inputstack[:,time_frame-7:time_frame+1,:, :]
            self.phase_stack = inputstack
    
    
        _,_,h1,h2 = self.phase_stack.shape
        
        self.x1 = (h1-self.cropsize)//self.stepsize+1
        self.x2 = (h2-self.cropsize)//self.stepsize+1

    
    def __len__(self):
        return self.x1*self.x2

    def __getitem__(self,index):
        p1 = index//self.x2
        p2 = index%self.x2

        x1 = self.stepsize*p1
        x2 = self.stepsize*p2

        input = self.phase_stack[:,:,x1:x1+self.cropsize,x2:x2+self.cropsize]
        input = np.nan_to_num(input)
        input[0,:,:,:] = (input[0,:,:,:] - np.mean(input[0,:,:,:])) / (np.std(input[0,:,:,:]) + 1e-8)
        input[1,:,:,:] = (input[1,:,:,:] - np.mean(input[1,:,:,:])) / (np.std(input[1,:,:,:]) + 1e-8)

        if self.input_transform:
            input = input.copy()
            input = self.input_transform(input)

        return input, index+1

def get_data_set(in_channels, in_frames, inputstack, time_frame, stepsize=20, cropsize=100,input_transform=None,well_num=1):
    return DataFolder(in_channels, in_frames, inputstack, time_frame, stepsize, cropsize, input_transform, well_num)


if __name__ == '__main__':
    inputfolders = [
        r'Y:\TFT_based_PFU\TFT_PFU_Network_Code_Yuzhu\Github\Example_Data\Input',
    ]
        
    for inputfolder in inputfolders:
        modelPath = './model_detection_epoch277.pth'

        time_start = time.time()
        diff_stack = h5.File(inputfolder + '\diff_stack_multiparas.mat', 'r')['diff_stack_multiparas'] # time * width *height * 2
        diff_stack = np.transpose(np.asarray(diff_stack), (3,0,2,1)) # 2 * time * height * width
        time_end = time.time()
        print('time loading raw data', time_end-time_start, 's')
        
        for timepoint in range(7, np.shape(diff_stack)[1],1): #(strat time: end time: timestep)

            log_folder = Path(os.path.join('./Example_Data/Output','log_20251121_8diffchannel_timeframe8_32_24_b3363_dr0.5_center50')).mkdir(parents=True,exist_ok=True)
            result_file = os.path.join('./Example_Data/Output','log_20251121_8diffchannel_timeframe8_32_24_b3363_dr0.5_center50','result_time'+str(timepoint+1)+'_epoch277.txt')
            result_file = open(result_file,'w')

            print('loading model')
            model = network.DenseNet(in_channels = 2, in_frames = 8, init_channels = 32, growth_rate = 24, blocks = [3,3,6,3], num_classes = 2, drop_rate = 0.5, bn_size = 16, batch_norm = True)
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            
            model.to(device)
            model.load_state_dict(torch.load(modelPath))
            model.eval()

            print('loading input data')   
            time_start = time.time()
            dataset = get_data_set(in_channels = 2, in_frames = 8, inputstack = diff_stack, time_frame = timepoint, stepsize = 10, cropsize = 50) 
            data_loader = DataLoader(dataset=dataset, num_workers=3,
                                batch_size=100, shuffle=False, drop_last=False,
                                pin_memory=True)
            time_end = time.time()
            print('time loading dataset', time_end-time_start, 's')

            time_start = time.time()
            first_line = True
            with torch.no_grad():
                for i,batch in enumerate(tqdm(data_loader)):
                    input = batch[0].float().to(device)
                    indices = batch[1]
                    output = model(input)

                    _,predicted = torch.max(output,1)
                    for j,lab in enumerate(predicted):
                        score = torch.exp(output[j,:]) / (torch.exp(output[j,0])+torch.exp(output[j,1]))
                        if first_line:
                            result_file.write(f'{indices[j]} {score[0]} {score[1]}')
                            first_line = False
                        else:
                            result_file.write(f'\n{indices[j]} {score[0]} {score[1]}')

            result_file.close()
            time_end = time.time()
            print('time cost', time_end-time_start, 's')
