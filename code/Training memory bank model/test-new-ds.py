# import comet_ml
from comet_ml import Experiment
import io
import torch
from sklearn.utils import class_weight
import imageio.v3 as iio
import torch.nn as nn
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
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import os
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import monai
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

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('--train_csv', type=str, help='')
parser.add_argument('--val_csv', type=str, help='')
parser.add_argument('--mount_point', type=str, help='')


parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=10, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=4, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=500, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=16, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')

parser.add_argument('--save_dir', default='.', type=str, help='directory to save models, runs')


args = parser.parse_args()

gpu_usg = args.gpu
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))
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

def check_dir_exist(path_dir):
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)

import multiprocessing as mp
import ctypes



class HystDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="vid_path", len_video = 'len_video',
                 class_column='class'):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.len_video = len_video

        self.num_sequences = (self.cumulative_frames[-1] - 9*len(df[self.img_column]))
        self.cumulative_frames = df[self.len_video].cumsum().to_list()


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
        return img[start_frame_idx:start_frame_idx+sequence_length,:,:,:]


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
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
        self.fc = nn.Linear(512, 6) ## change to number of classes 
        self.dropout = nn.Dropout(p=0.2)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224) ## change to input image size
        x = self.share.forward(x)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc(y)
        return y

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

def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)): #for each video
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)): # for second 0 to seconds N -10
            idx.append(j) # add each second of videos 
        count += list_each_length[i]
    return idx
    
def get_csv_data(data_path, mount_point, split='train'):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1)
    df = df.head(50)

    df['vid_path'] = mount_point + df['vid_path'].astype(str)

    labels = df['class'].to_list()

    train_transforms = TrainTransforms()
    test_transforms = TestTransforms()

    if split=='train':
        dataset = HystDataset(df, mount_point, train_transforms)
    elif split=='val':
        dataset = HystDataset(df, mount_point, test_transforms)

    return dataset, labels


# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

sig_f = nn.Sigmoid()


def create_frames_index(sequence_length, list_start_idx):
    list_idx = []
    for idx_frame in list_start_idx:
        for j in range(sequence_length):
            list_idx.append(idx_frame + j)
    return list_idx

def train_model():
    # comet

    exp = Experiment(api_key='jvo9wdLqVzWla60yIWoCd0fX2',
                        project_name='TMRnet',
                        workspace='luciedle')
    
    df_train = pd.read_csv(args.train_csv)    

    unique_classes = np.sort(np.unique(df_train['class']))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train['class']))    
    unique_class_weights = torch.tensor(unique_class_weights).to(torch.float32)

    train_dataset, train_labels= get_csv_data(args.train_csv, args.mount_point)
    val_dataset, val_labels = get_csv_data(args.val_csv, args.mount_point, 'valid')


    
    save_dir =args.save_dir

    run_dir = save_dir + '/runs/' + str(args.lr) + '/'
    check_dir_exist(run_dir)
    writer = SummaryWriter(run_dir)

    train_num_frames = train_dataset.num_sequences
    num_train_all = train_dataset.num_sequences
    num_val_all =val_dataset.num_sequences
    train_we_use_start_idx = get_useful_start_idx(sequence_length, np.arange(train_num_frames))
    
    # print('num train start idx: {:6d}'.format(len(train_we_use_start_idx)))
    print('length of train dataset: {:6d}'.format(len(train_dataset)))

    disp_iter = 100

    ## val dataset = 15041
    ## all frames to use
    val_loader = DataLoader(
        val_dataset, ## = number of frames use, 15041
        batch_size=args.val,
        num_workers=workers,
        pin_memory=False,
        shuffle=True
    )

    model = resnet_lstm()
    model = DataParallel(model)
    model.to(device)

    criterion_phase = nn.CrossEntropyLoss(reduction='sum', weight=unique_class_weights.to(device))

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
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        np.random.shuffle(train_we_use_start_idx)

        train_loader_80 = DataLoader(
            train_dataset,
            batch_size=args.train,
            num_workers=workers,
            shuffle=True
        )

        # Sets the module in training mode.
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()
        
        for i, data  in enumerate(train_loader_80):
            optimizer.zero_grad()

            inputs, labels_phase = data[0].to(device), data[1].to(device)

            outputs_phase = model.forward(inputs)
            outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            loss = loss_phase
            loss.backward()
            optimizer.step()

            running_loss_phase += loss_phase.data.item()
            train_loss_phase += loss_phase.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

            if i % disp_iter == 0: ## every 10
                # ...log the running loss
                
                running_loss = running_loss_phase / (float(train_batch_size*disp_iter))

                # ...log the training acc
                running_acc = float(minibatch_correct_phase) / (float(train_batch_size)*disp_iter)

                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            if (i+1)*train_batch_size >= num_train_all:               
                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            batch_progress += 1
            if batch_progress*train_batch_size >= num_train_all:
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', num_train_all, num_train_all), end='\n')
            else:
                percent = round(batch_progress*train_batch_size / num_train_all * 100, 2)
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress*train_batch_size, num_train_all), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / float(num_train_all) * sequence_length
        train_average_loss_phase = train_loss_phase / num_train_all * sequence_length

        # Sets the module in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_phase = []
        val_all_labels_phase = []

        print("####### training done. Starting evaluation... ############")

        with torch.no_grad():
            for data in val_loader:

                # if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)

                outputs_phase = model.forward(inputs)
                outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                # TODO

                for pred in preds_phase:
                    val_all_preds_phase.append(int(pred.data.cpu()))
                for label in labels_phase:
                    val_all_labels_phase.append(int(label.data.cpu()))

                val_progress += 1
                if val_progress*val_batch_size >= num_val_all:
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', num_val_all, num_val_all), end='\n')
                else:
                    percent = round(val_progress*val_batch_size / num_val_all * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*val_batch_size, num_val_all), end='\r')

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / float(num_val_all) ## or num_val_frames ?
        val_average_loss_phase = val_loss_phase / num_val_all

        val_recall_phase = metrics.recall_score(val_all_labels_phase,val_all_preds_phase, average='macro')
        val_precision_phase = metrics.precision_score(val_all_labels_phase,val_all_preds_phase, average='macro')
        val_precision_each_phase = metrics.precision_score(val_all_labels_phase,val_all_preds_phase, average=None)
        val_recall_each_phase = metrics.recall_score(val_all_labels_phase,val_all_preds_phase, average=None)


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


        train_metrics = {"training loss": running_loss, 
                         "traininc acc": running_acc,
                         
                         "valid loss(phase)":val_average_loss_phase,
                         "valid accu(phase)":val_accuracy_phase,

                         "val_precision_each_phase": np.mean(val_precision_each_phase),
                         "val_recall_each_phase": np.mean(val_recall_each_phase),

                         "val_precision_phase": val_precision_phase,
                         "val_recall_phase": val_recall_phase,
        }
        exp.log_metrics(train_metrics, epoch=epoch)

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
                     + "_length_" + str(sequence_length) \
                     + "_opt_" + str(optimizer_choice) \
                     + "_mulopt_" + str(multi_optim) \
                     + "_flip_" + str(use_flip) \
                     + "_crop_" + str(crop_type) \
                     + "_batch_" + str(train_batch_size) \
                     + "_train_" + str(save_train_phase) \
                     + "_val_" + str(save_val_phase)



        model_dir = save_dir + "/best_model/" + str(args.lr) + "/"
        check_dir_exist(model_dir)

        torch.save(best_model_wts, model_dir+base_name+".pth")
        print("best_epoch",str(best_epoch))

        all_models_dir = save_dir
        torch.save(model.module.state_dict(), all_models_dir+ "/latest_model_"+str(epoch)+".pth")


if __name__ == "__main__":
    train_model()

print('Done')
print()