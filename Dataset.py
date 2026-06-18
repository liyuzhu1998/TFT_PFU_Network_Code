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
import scipy.io

def is_target_file(filename):
    return filename.endswith(".mat")


def load_img(filepath):
    y = scipy.io.loadmat(filepath)
    return y


class DataFolder(data.Dataset):

    def __init__(self,in_channels,in_frames ,image_dir, lab, start_frame = 0, input_transform=None, tf_crop=False):
        super(DataFolder,self).__init__()

        self.in_channels = in_channels
        self.in_frames = in_frames
        self.start_frame = start_frame
        cur_label = np.asarray(lab)
        
        self.image_filenames = image_dir
        self.label = [cur_label]*len(self.image_filenames) # expand one item to #image_filenames items
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self,index):
        path = self.image_filenames[index]
        s = self.start_frame
        try:
            input_raw = load_img(path)
        except:
            print('Error loading file:', path)
        input_raw = input_raw['input'].astype('float32')
        if self.in_frames == 1:
            input = input_raw[[2,3,4],s:self.in_frames+s,25:75,25:75].astype(float)
        elif self.in_frames == 2:
            input = input_raw[[2,3,4],4:8:3,25:75,25:75].astype(float)
            #input = input_raw[[4],4:8:3,25:75,25:75].astype(float) # 0: intensity 1: amplitude 2: phase 3: diff-1 4: diff-2
        elif self.in_frames == 8:
            input = input_raw[[2,3,4],:,25:75,25:75].astype(float)
        label = self.label[index]

        input = np.nan_to_num(input)
        input[0,:,:,:] = (input[0,:,:,:] - np.mean(input[0,:,:,:])) / (np.std(input[0,:,:,:]) + 1e-8)
        input[1,:,:,:] = (input[1,:,:,:] - np.mean(input[1,:,:,:])) / (np.std(input[1,:,:,:]) + 1e-8)
        input[2,:,:,:] = (input[2,:,:,:] - np.mean(input[2,:,:,:])) / (np.std(input[2,:,:,:]) + 1e-8)
        # input = (input - np.min(input)) / (np.max(input) - np.min(input))
        # input = input * 2 - 1
        
        if random.randint(0,1):
            input = np.flip(input,axis=3).copy()
        if random.randint(0,1):
            input = np.flip(input,axis=2).copy()
        rot = random.randint(0,3)
        img = np.rot90(input, k=rot, axes=(2,3)).copy()

        return img,label,path

def get_data_set(in_channels,in_frames,image_dir,lab,start_frame = 0, input_transform=None, tf_crop=False):
    return DataFolder(in_channels,in_frames,image_dir,lab,start_frame,input_transform,tf_crop)