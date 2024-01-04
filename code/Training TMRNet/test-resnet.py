from comet_ml import Experiment
import io
import torch
import torch.nn as nn
from sklearn.utils import class_weight

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
import os
from sklearn import metrics
from NLBlock_MutiConv6_3 import NLBlock
from NLBlock_MutiConv6_3 import TimeConv
from torchvision.models import resnet50, ResNet50_Weights
import imageio
import pandas as pd
import monai
import matplotlib.pyplot as plt
from monai.transforms import (
    EnsureChannelFirst,
    ToTensor,
    ScaleIntensityRange,
    RandGaussianNoise,
    Compose,
    RandRotated,
    ScaleIntensityd,    
    ToTensord,
    EnsureType,
    Activations, 
    AsDiscrete, 
    Resized,
    RandZoomd,
    Lambdad,
    CenterSpatialCrop,
    ResizeWithPadOrCrop
)

import imageio.v3 as iio

parser = argparse.ArgumentParser(description='lstm training')

parser.add_argument('--train_csv', type=str, help='')
parser.add_argument('--val_csv', type=str, help='')
parser.add_argument('--mount_point', type=str, help='')


parser.add_argument('--load_LFB', default=True, type=bool, help='whether load exist long term feature bank')
parser.add_argument('--model_path', default='./LFB/FBmodel/x.pth', type=str, help='the path of the memory bank model')

parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=90, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=50, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='EPOCHS to train and val, default 25')
parser.add_argument('-w', '--work', default=12, type=int, help='num of WORKERS to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-7, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--LFB_l', default=30, type=int, help='long term feature bank length')

args = parser.parse_args()

gpu_usg = args.gpu
SEQ_LENGTH = args.seq
TRAIN_BS = args.train
VAL_BS = args.val
optimizer_choice = args.opt
multi_optim = args.multi
EPOCHS = args.epo
WORKERS = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

LFB_lENGTH = args.LFB_l
LOAD_EXIST_LFB = args.load_LFB

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(SEQ_LENGTH))
print('train batch size: {:6d}'.format(TRAIN_BS))
print('valid batch size: {:6d}'.format(VAL_BS))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(EPOCHS))
print('num of workers  : {:6d}'.format(WORKERS))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
print('learning rate   : {:.4f}'.format(learning_rate))
print('momentum for sgd: {:.4f}'.format(momentum))
print('weight decay    : {:.4f}'.format(weight_decay))
print('dampening       : {:.4f}'.format(dampening))
print('use nesterov    : {:6d}'.format(use_nesterov))
print('method for sgd  : {:6d}'.format(sgd_adjust_lr))
print('step for sgd    : {:6d}'.format(sgd_step))
print('gamma for sgd   : {:.4f}'.format(sgd_gamma))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class HystDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="vid_path", len_video = 'len_video',
                 class_column='class'):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.len_video = len_video

        self.cumulative_frames = df[self.len_video].cumsum().to_list()
        self.num_sequences = (self.cumulative_frames[-1] - 9*len(df[self.img_column]))


        # Keep a cache of samples already read
        # self.cache = {}

        # shared_array_base = mp.Array(ctypes.c_float, self.num_sequences*c*h*w)
        # shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        # shared_array = shared_array.reshape(nb_samples, c, h, w)
        # self.shared_array = torch.from_numpy(shared_array)
        # self.use_cache = False


    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx): ## idx should be the sequence index
        video_idx, start_frame_idx = self.find_video_and_frame(idx)

        vid_path = os.path.join(self.mount_point, self.df.iloc[video_idx][self.img_column])
        try:
            sequence = self.read_sequence(vid_path, start_frame_idx)
        except Exception as e:
            print("error reading sequence")
            sequence = np.zeros((256, 256, 3), dtype=np.float32)

        if self.transform:
            sequence = self.transform(sequence)

        if self.class_column != None:
            class_label = torch.tensor(self.df.iloc[video_idx][self.class_column]).to(torch.long)

        return sequence, class_label

    def find_video_and_frame(self, idx):
        # Find the video number from the global sequence idx
        video_idx = 0
        for i, total_frames in enumerate(self.cumulative_frames):
            if total_frames > idx:
                video_idx = i - 1
                break

        start_frame_idx = idx - self.cumulative_frames[video_idx] ## remove the previous videos length to get the indexing from 0 to len_video

        return video_idx, start_frame_idx

    def read_sequence(self, vid_path, start_frame_idx):

        # if vid_path and start_frame_idx in cache
        # return cache

        # else
        img = iio.imread(vid_path, plugin="pyav")
        return img[start_frame_idx:start_frame_idx+SEQ_LENGTH,:,:,:]


class TrainTransforms:
    def __init__(self, height: int = 224, num_frames=10):
        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                # RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), antialias=True),                
                ToTensor(dtype=torch.float32),                
                ScaleIntensityRange(0, 255, 0, 1)
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)

class EvalTransforms:
    def __init__(self, height: int = 224, num_frames=10):
        self.test_transform = transforms.Compose(
            [
                # RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                ResizeWithPadOrCrop((-1, 224, 224)),
                ToTensor(dtype=torch.float32),                
                ScaleIntensityRange(0, 255, 0, 1)                
            ]
        )
    def __call__(self, inp):
        return self.test_transform(inp)

class TestTransforms:
    def __init__(self, height: int = 224, num_frames=512):
        self.test_transform = transforms.Compose(
            [
                # RandomChoice(num_frames=num_frames),
                EnsureChannelFirst(channel_dim=-1),
                PermuteTimeChannel(),
                ResizeWithPadOrCrop((-1, 224, 224)),
                ToTensor(dtype=torch.float32),                
                ScaleIntensityRange(0, 255, 0, 1)                
            ]
        )
    def __call__(self, inp):
        return self.test_transform(inp)

class PermuteTimeChannel:
    def __init__(self, permute=(1,0,2,3)):
        self.permute = permute        
    def __call__(self, x):
        return torch.permute(x, self.permute)

class RandomChoice:
    def __init__(self, num_frames=-1):
        self.num_frames = num_frames
    def __call__(self, x):
        if self.num_frames > 0:
            # idx = torch.randint(0, high=x.shape[0], size=(min(self.num_frames, len(x)),))
            idx = torch.randint(0, high=x.shape[0], size=(self.num_frames,))
            if self.num_frames == 1:
                x = x[idx[0]]
            else:
                idx, _ = torch.sort(idx)
                x = x[idx]
        return x
    
def display_sequence(video):
    batch_size =10
    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)

    video = video.data.cpu()
    for idx in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, idx+1)
        img = video[idx,:,:,:]


        # img = np.reshape(img, (224,224,3))
        img = np.transpose(img,axes=(1,2,0))
        
        x_norm = (img-np.min(img))/(np.max(img)-np.min(img))        
        plt.imshow(x_norm[:,:,1])
    # plt.savefig(path)
    plt.close()
    return fig 

class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # resnet = resnet50(pretrained=True)

        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc_c = nn.Linear(512, 7)
        self.fc_h_c = nn.Linear(1024, 512)
        self.nl_block = NLBlock()
        self.dropout = nn.Dropout(p=0.5)
        self.time_conv = TimeConv()

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc_c.weight)
        init.xavier_uniform_(self.fc_h_c.weight)

    def forward(self, x, long_feature):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, SEQ_LENGTH, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = y[SEQ_LENGTH - 1::SEQ_LENGTH]

        Lt = self.time_conv(long_feature)

        y_1 = self.nl_block(y, Lt)
        y = torch.cat([y, y_1], dim=1)
        y = self.dropout(self.fc_h_c(y))
        y = F.relu(y)
        y = self.fc_c(y)
        return y


class resnet_lstm_LFB(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_LFB, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # resnet = resnet50(pretrained=True)

        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, SEQ_LENGTH, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = y[SEQ_LENGTH - 1::SEQ_LENGTH]
        return y


def get_useful_start_idx(SEQ_LENGTH, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - SEQ_LENGTH)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_long_feature(start_index_list, dict_start_idx_LFB, lfb):
    long_feature = []

    ## lfb = the pickle we created
    for j in range(len(start_index_list)): ## for each starting index (video)
        long_feature_each = []
        
        last_LFB_index_no_empty = dict_start_idx_LFB[int(start_index_list[j])]
        
        for k in range(LFB_lENGTH): ## from 0 to 29
            LFB_index = int(start_index_list[j] - k - 1) # -1 to -30
            if LFB_index in dict_start_idx_LFB:   
                LFB_index = dict_start_idx_LFB[LFB_index]
                long_feature_each.append(lfb[LFB_index])
                last_LFB_index_no_empty = LFB_index
            else:
                long_feature_each.append(lfb[last_LFB_index_no_empty])
            
        long_feature.append(long_feature_each)

    return long_feature



def get_long_feature_v2(start_index_list, dict_start_idx_LFB, lfb):
    long_feature = []
    
    for start_index in start_index_list: ## for each starting index (video)
        long_feature_each = []
        last_LFB_index_no_empty = dict_start_idx_LFB[start_index]
        
        for k in range(LFB_lENGTH):
            LFB_index = (start_index - k - 1)
            if int(LFB_index) in dict_start_idx_LFB:                
                LFB_index = dict_start_idx_LFB[int(LFB_index)]
                long_feature_each.append(lfb[LFB_index])
                last_LFB_index_no_empty = LFB_index
            else:
                long_feature_each.append(lfb[last_LFB_index_no_empty])
            
        long_feature.append(long_feature_each)
    return long_feature

def get_csv_data(data_path, mount_point, split='train'):
    df = pd.read_csv(data_path)

    df['vid_path'] = mount_point + df['vid_path'].astype(str)

    train_transforms = TrainTransforms()
    test_transforms = TestTransforms()

    if split=='train':
        dataset = HystDataset(df, mount_point, train_transforms)
    elif split=='val':
        dataset = HystDataset(df, mount_point, test_transforms)
    elif split=='LFB':
        dataset = HystDataset(df, mount_point, test_transforms)

    return dataset, df['len_video']

sig_f = nn.Sigmoid()

# Long Term Feature bank
g_LFB_train = np.zeros(shape=(0, 512))
g_LFB_val = np.zeros(shape=(0, 512))


def create_frames_index(sequence_length, list_start_idx):
    list_idx = []
    for idx_frame in list_start_idx:
        for j in range(sequence_length):
            list_idx.append(idx_frame + j)
    return list_idx

def create_mapping_feature_frames(n_frames, len_videos):

    prev_video_len = 0
    dic_values = []
    for vid_len in len_videos :
        # print(prev_video_len, prev_video_len+vid_len-10)
        for j in range(prev_video_len, prev_video_len + vid_len -9):
            dic_values.append(j)
        prev_video_len += vid_len
        
    idx, val = zip(*list(enumerate(dic_values)))
    mapping_features = dict(zip(idx, val))

    return mapping_features




def train_model():
    # TensorBoard
    exp = Experiment(api_key='jvo9wdLqVzWla60yIWoCd0fX2',
                        project_name='TMRnet_2nd_training',
                        workspace='luciedle')
    

    df_train = pd.read_csv(args.train_csv)    

    unique_classes = np.sort(np.unique(df_train['class']))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train['class']))    
    unique_class_weights = torch.tensor(unique_class_weights).to(torch.float32)

    train_dataset, train_list_len_vids = get_csv_data(args.train_csv, args.mount_point)
    val_dataset, val_list_len_vids = get_csv_data(args.val_csv, args.mount_point, 'val')
    lfb_dataset, _ = get_csv_data(args.val_csv, args.mount_point, 'LFB')

    map_features_frames = create_mapping_feature_frames(train_dataset.cumulative_frames[-1], train_list_len_vids.to_list())

    ## both are tghe same as above
    # train_idx_LFB = create_frames_index(SEQ_LENGTH,train_we_use_start_idx_80_LFB)
    # val_idx_LFB = create_frames_index(SEQ_LENGTH,val_we_use_start_idx_LFB)


    """
        dictionnary insight :
        {1:1, ... 1298:1298, 1308:1299}

        index = index
        value = index of frames

        when we reach the end of the first video at the frame 1298 (1298:1298). 
        we add +9 to start a new video to avoid taking the previous frame index 
    
    """
    num_train_all = train_dataset.num_sequences
    num_val_all = val_dataset.num_sequences

    # print('num train start idx 80: {:6d}'.format(len(train_we_use_start_idx_80_LFB)))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num of all valid use: {:6d}'.format(num_val_all))
    # print('num of all train LFB use: {:6d}'.format(len(train_idx_LFB)))
    # print('num of all valid LFB use: {:6d}'.format(len(val_idx_LFB)))

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BS,
        # sampler=SeqSampler(val_dataset, val_idx_LFB),
        num_workers=WORKERS,
        pin_memory=False
    )

    global g_LFB_train
    global g_LFB_val
    print("loading features!>.........")

    #LOAD_EXIST_LFB = False
    if not LOAD_EXIST_LFB:

        print("no LFB pickle ! Generating...")        
        train_feature_loader = DataLoader(
            train_dataset_80_LFB,
            batch_size=VAL_BS,
            # sampler=SeqSampler(train_dataset_80_LFB, train_idx_LFB),
            num_workers=WORKERS,
            pin_memory=False
        )
        val_feature_loader = DataLoader(
            val_dataset,
            batch_size=VAL_BS,
            sampler=SeqSampler(val_dataset, val_idx_LFB),
            num_workers=WORKERS,
            pin_memory=False
        )


        model_LFB = resnet_lstm_LFB()

        # model_LFB = torch.compile(model_LFB)

        model_LFB.load_state_dict(torch.load(args.model_path), strict=False)

        if use_gpu:
            model_LFB = DataParallel(model_LFB)
            model_LFB.to(device)

        for params in model_LFB.parameters():
            params.requires_grad = False

        model_LFB.eval()

        with torch.no_grad():

            for data in train_feature_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]
                print(inputs.shape)

                inputs = inputs.view(-1, SEQ_LENGTH, 3, 224, 224)
                print(inputs.shape) ## same outputs [ n_vid, 10, 3, 224, 224]
                outputs_feature = model_LFB.forward(inputs)
                print(outputs_feature.shape)

                for j in range(len(outputs_feature)):
                    save_feature = outputs_feature.data.cpu()[j].numpy()
                    save_feature = save_feature.reshape(1, 512)
                    g_LFB_train = np.concatenate((g_LFB_train, save_feature),axis=0)

                print("train feature length:",len(g_LFB_train))

            for data in val_feature_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                inputs = inputs.view(-1, SEQ_LENGTH, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs)

                for j in range(len(outputs_feature)):
                    save_feature = outputs_feature.data.cpu()[j].numpy()
                    save_feature = save_feature.reshape(1, 512)
                    g_LFB_val = np.concatenate((g_LFB_val, save_feature), axis=0)

                print("val feature length:",len(g_LFB_val))

        print("finish!")
        g_LFB_train = np.array(g_LFB_train)
        g_LFB_val = np.array(g_LFB_val)

        # with open("./LFB/g_LFB_train.pkl", 'wb') as f:
        #     pickle.dump(g_LFB_train, f)

        # with open("./LFB/g_LFB_val.pkl", 'wb') as f:
        #     pickle.dump(g_LFB_val, f)
    
    else:
        with open("./LFB/g_LFB_train.pkl", 'rb') as f:
            g_LFB_train = pickle.load(f)

        with open("./LFB/g_LFB_val.pkl", 'rb') as f:
            g_LFB_val = pickle.load(f)

        print("load completed")

    print("g_LFB_train shape:",g_LFB_train.shape)
    print("g_LFB_val shape:",g_LFB_val.shape)
    
    torch.cuda.empty_cache()

    model = resnet_lstm()

    model.load_state_dict(torch.load(args.model_path), strict=False)
       
    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    criterion_phase = nn.CrossEntropyLoss(reduction='sum')

    optimizer = None
    exp_lr_scheduler = None

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters()},
                {'params': model.module.time_conv.parameters(), 'lr': learning_rate},
                {'params': model.module.nl_block.parameters(), 'lr': learning_rate},
                {'params': model.module.fc_h_c.parameters(), 'lr': learning_rate},
                {'params': model.module.fc_c.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters()},
                {'params': model.module.time_conv.parameters(), 'lr': learning_rate},
                {'params': model.module.nl_block.parameters(), 'lr': learning_rate},
                {'params': model.module.fc_h_c.parameters(), 'lr': learning_rate},
                {'params': model.module.fc_c.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        # np.random.shuffle(train_we_use_start_idx_80_LFB)
        # train_idx_80 = create_frames_index(SEQ_LENGTH, train_we_use_start_idx_80_LFB)

        train_loader_80 = DataLoader(
            train_dataset,
            batch_size=TRAIN_BS,
            # sampler=SeqSampler(train_dataset_80, train_idx_80),
            num_workers=WORKERS,
            pin_memory=False
        )

        # Sets the module in training mode.
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()

        for i, data in enumerate(train_loader_80):
            """
                data: list of 3 elements
                data[0], our sequence of frames
                data[1], corresponding class 
                data[2], corresponding frame index 
            
            """
            
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]


            labels_phase = labels_phase[(SEQ_LENGTH - 1)::SEQ_LENGTH]

            start_index_list = data[2]

            start_index_list = start_index_list[0::SEQ_LENGTH] ## take index of each video
            long_feature = get_long_feature(start_index_list=start_index_list,
                                            dict_start_idx_LFB=map_features_frames,
                                            lfb=g_LFB_train) #g_LFB_train of shape 70,744 - 512
            
            long_feature = np.array(long_feature)
            # print(long_feature.shape) # (10, 30, 512)

            long_feature = (torch.Tensor(long_feature)).to(device)

            print(inputs.shape)
            inputs = inputs.view(-1, SEQ_LENGTH, 3, 224, 224)
            # TODO


            for vid in range(inputs.shape[0]):
                img = display_sequence(inputs[vid,...])

                buf = io.BytesIO()
                img.savefig(buf, format='png')
                buf.seek(0)
                vid_name = "video number: " + str(vid)
                exp.log_image(buf, vid_name,step=epoch)

            buf.close()


            outputs_phase = model.forward(inputs, long_feature=long_feature)
            print(outputs_phase.shape)

            _, preds_phase = torch.max(outputs_phase.data, 1)
            print(preds_phase.shape)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            loss = loss_phase
            loss.backward()
            optimizer.step()

            running_loss_phase += loss_phase.data.item()
            train_loss_phase += loss_phase.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase


            if i % 500 == 499:
                # ...log the running loss
                batch_iters = epoch * num_train_all/SEQ_LENGTH + i*TRAIN_BS/SEQ_LENGTH
                writer.add_scalar('training loss phase',
                                  running_loss_phase / (TRAIN_BS*500/SEQ_LENGTH) , batch_iters)
                # ...log the training acc
                writer.add_scalar('training acc phase',
                                  float(minibatch_correct_phase) / (float(TRAIN_BS)*500/SEQ_LENGTH), batch_iters)
                # ...log the val acc loss

                # val_loss_phase, val_corrects_phase = valMinibatch(val_loader, model, dict_start_idx_LFB=dict_val_start_idx_LFB)
                # writer.add_scalar('validation acc miniBatch phase',
                #                   float(val_corrects_phase) / float(num_val_we_use), batch_iters)
                # writer.add_scalar('validation loss miniBatch phase',
                #                   float(val_loss_phase) / float(num_val_we_use), batch_iters)

                # val_accuracy_phase = float(val_corrects_phase) / float(num_val_we_use)

                # if val_accuracy_phase > best_val_accuracy_phase:
                #     save_val_phase = int("{:4.0f}".format(val_accuracy_phase * 10000))
                #     public_name = "minibatch_cnn_lstm_phase_valPhase_" + str(save_val_phase)

                #     torch.save(model.module.state_dict(),"./best_model/non-local/pretrained_lr5e-7_L30_2fc_copy_mutiConv6_3_v2/" + public_name + ".pth")

                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            if (i+1)*TRAIN_BS >= num_train_all:               
                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            batch_progress += 1
            if batch_progress*TRAIN_BS >= num_train_all:
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', num_train_all, num_train_all), end='\n')
            else:
                percent = round(batch_progress*TRAIN_BS / num_train_all * 100, 2)
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress*TRAIN_BS, num_train_all), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / float(num_train_all) * SEQ_LENGTH
        train_average_loss_phase = train_loss_phase / num_train_all * SEQ_LENGTH

        # Sets the module in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_phase = []
        val_all_labels_phase = []

        with torch.no_grad():
            for data in val_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                labels_phase = labels_phase[(SEQ_LENGTH - 1)::SEQ_LENGTH]

                start_index_list = data[2]
                start_index_list = start_index_list[0::SEQ_LENGTH]
                long_feature = get_long_feature(start_index_list=start_index_list,
                                                dict_start_idx_LFB=dict_val_start_idx_LFB,
                                                lfb=g_LFB_val)

                long_feature = np.array(long_feature)
                # print(long_feature.shape) # (10, 30, 512)
                long_feature = torch.Tensor(long_feature).to(device)

                inputs = inputs.view(-1, SEQ_LENGTH, 3, 224, 224)
                outputs_phase = model.forward(inputs, long_feature=long_feature)
                # outputs_phase = outputs_phase[SEQ_LENGTH - 1::SEQ_LENGTH]

                print(outputs_phase.shape)
                _, preds_phase = torch.max(outputs_phase.data, 1)
                print(preds_phase.shape)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                # TODO
                ## can i send directy the tensor to cpu ?
                for pred in preds_phase:
                    val_all_preds_phase.append(pred.cpu())
                for label in labels_phase:
                    val_all_labels_phase.append(label.cpu())


                val_progress += 1
                if val_progress*VAL_BS >= num_val_all:
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', num_val_all, num_val_all), end='\n')
                else:
                    percent = round(val_progress*VAL_BS / num_val_all * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*VAL_BS, num_val_all), end='\r')

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / float(num_val_we_use)
        val_average_loss_phase = val_loss_phase / num_val_we_use

        val_precision_each_phase = metrics.precision_score(val_all_labels_phase,val_all_preds_phase, average=None)
        val_recall_each_phase = metrics.recall_score(val_all_labels_phase,val_all_preds_phase, average=None)

        writer.add_scalar('validation acc epoch phase',
                          float(val_accuracy_phase),epoch)
        writer.add_scalar('validation loss epoch phase',
                          float(val_average_loss_phase),epoch)

        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss(phase): {:4.4f}'
              ' train accu(phase): {:.4f}'
              ' valid in: {:2.0f}m{:2.0f}s'
              ' valid loss(phase): {:4.4f}'
              ' valid accu(phase): {:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss_phase,
                      train_accuracy_phase,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss_phase,
                      val_accuracy_phase))


        print("val_precision_each_phase:", val_precision_each_phase)
        print("val_recall_each_phase:", val_recall_each_phase)

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss_phase)

        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch = epoch
        if val_accuracy_phase == best_val_accuracy_phase:
            if train_accuracy_phase > correspond_train_acc_phase:
                correspond_train_acc_phase = train_accuracy_phase
                best_model_wts = copy.deepcopy(model.module.state_dict())
                best_epoch = epoch

        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        base_name = "lstm" \
                     + "_epoch_" + str(best_epoch) \
                     + "_length_" + str(SEQ_LENGTH) \
                     + "_opt_" + str(optimizer_choice) \
                     + "_mulopt_" + str(multi_optim) \
                     + "_flip_" + str(use_flip) \
                     + "_crop_" + str(crop_type) \
                     + "_batch_" + str(TRAIN_BS) \
                     + "_train_" + str(save_train_phase) \
                     + "_val_" + str(save_val_phase)

        torch.save(best_model_wts, "./best_model/non-local/pretrained_lr5e-7_L30_2fc_copy_mutiConv6_3_v2/"+base_name+".pth")
        print("best_epoch",str(best_epoch))

        torch.save(model.module.state_dict(), "./temp/non-local/pretrained_lr5e-7_L30_2fc_copy_mutiConv6_3_v2/latest_model_"+str(epoch)+".pth")


def main():
    train_model()

if __name__ == "__main__":
    main()

print('Done')
print()
