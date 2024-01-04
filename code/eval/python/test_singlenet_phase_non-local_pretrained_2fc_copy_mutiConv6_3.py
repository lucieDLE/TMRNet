import torch
import pdb
import torch.nn as nn
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import DataParallel
import torch.nn.functional as F
import os
from PIL import Image
import time
import pickle
import numpy as np
import argparse
from torchvision.transforms import Lambda
from NLBlock_MutiConv6_3 import NLBlock
from NLBlock_MutiConv6_3 import TimeConv
import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

def print_params(args):
    '''
    gpu_usg = ",".join(list(map(str, args.gpu)))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
    '''

    model_pure_name, _ = os.path.splitext(args.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('gpu             : ', device)
    print('sequence length : {:6d}'.format(args.seq))
    print('test batch size : {:6d}'.format(args.test))
    print('num of args.work  : {:6d}'.format(args.work))
    print('test crop type  : {:6d}'.format(args.crop))
    print('name of this model: {:s}'.format(args.model))  # so we can store all result in the same file
    print('Result store path: {:s}'.format(model_pure_name))

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
def get_csv_data(data_path, split='train'):
    df = pd.read_csv(data_path)

    paths = df['frame'].to_numpy()
    labels = df['class'].to_numpy()
    n_classes = np.unique(labels)
    num_each = []

    for class_name in n_classes:
        df_tmp = df.loc[df['class']==class_name]
        len_vids = df_tmp['id'].value_counts().to_list()
        for elt in len_vids:
           num_each.append(elt)

    print((num_each))
    print('{}_paths  : {:6d}'.format(split, len(paths)))

    train_transforms = None
    test_transforms = None

    if args.crop == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif args.crop == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif args.crop == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif args.crop == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])(crop) for crop in crops]))
        ])
    elif args.crop == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])(crop) for crop in crops]))
        ])
    dataset = CholecDataset(paths, labels, test_transforms)

    return dataset, num_each

class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels = self.file_labels[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels, index

    def __len__(self):
        return len(self.file_paths)

class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
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
        self.fc_c = nn.Linear(512, 6)
        self.fc_h_c = nn.Linear(1024, 512)
        self.nl_block = NLBlock()
        self.time_conv = TimeConv()

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc_c.weight)
        init.xavier_uniform_(self.fc_h_c.weight)

    def forward(self, x, long_feature=None):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, args.seq, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = y[args.seq - 1::args.seq]

        Lt = self.time_conv(long_feature)

        y_1 = self.nl_block(y, Lt)
        y = torch.cat([y, y_1], dim=1)
        y = self.fc_h_c(y)
        y = F.relu(y)
        y = self.fc_c(y)
        return y

class resnet_lstm_LFB(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm_LFB, self).__init__()
        resnet = models.resnet50(pretrained=True)
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
        x = x.view(-1, args.seq, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = y[args.seq - 1::args.seq]
        return y

def get_useful_start_idx(seq, list_each_length):
    idx = []
    count = 0
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - seq)):
            idx.append(j)
        count += list_each_length[i]
    return idx

def get_useful_start_idx_LFB(seq, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - seq)):
            idx.append(j)
        count += list_each_length[i]
    return idx

def get_long_feature(start_index_list, dict_start_idx_LFB, lfb):
    long_feature = []
    for j in range(len(start_index_list)):
        long_feature_each = []
        
        # 上一个存在feature的index
        last_LFB_index_no_empty = dict_start_idx_LFB[int(start_index_list[j])]
        
        for k in range(args.LFB_l):
            LFB_index = (start_index_list[j] - k - 1)
            if int(LFB_index) in dict_start_idx_LFB:                
                LFB_index = dict_start_idx_LFB[int(LFB_index)]
                long_feature_each.append(lfb[LFB_index])
                last_LFB_index_no_empty = LFB_index
            else:
                long_feature_each.append(lfb[last_LFB_index_no_empty])
                
        long_feature.append(long_feature_each)
    return long_feature

def get_test_data(data_path):
    with open(data_path, 'rb') as f:
        test_paths_labels = pickle.load(f)

    test_paths = test_paths_labels[0]
    test_labels = test_paths_labels[1]
    test_num_each = test_paths_labels[2]

    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    test_labels = np.asarray(test_labels, dtype=np.int64)

    test_transforms = None
    if args.crop == 0:
        test_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif args.crop == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif args.crop == 5:
        test_transforms = transforms.Compose([
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])(crop) for crop in crops]))
        ])
    elif args.crop == 10:
        test_transforms = transforms.Compose([
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])(crop) for crop in crops]))
        ])
    elif args.crop == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif args.crop == 3:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])

    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return test_dataset, test_num_each

class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

# Long Term Feature bank
g_LFB_test = np.zeros(shape=(0, 512))

def test_model(args):
    model_pure_name, _ = os.path.splitext(args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    df_test = pd.read_csv(args.csv)    

    unique_classes = np.sort(np.unique(df_test['class']))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_test['class']))    
    unique_class_weights = torch.tensor(unique_class_weights).to(torch.float32)


    test_dataset, test_num_each = get_csv_data(args.csv)

    num_test = len(test_dataset)

    test_useful_start_idx = get_useful_start_idx(args.seq, test_num_each)
    test_useful_start_idx_LFB = get_useful_start_idx_LFB(args.seq, test_num_each)

    num_test_we_use = len(test_useful_start_idx)
    num_test_we_use_LFB = len(test_useful_start_idx_LFB)

    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]
    test_we_use_start_idx_LFB = test_useful_start_idx_LFB[0:num_test_we_use_LFB]

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(args.seq):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_test_all = len(test_idx)

    test_idx_LFB = []
    for i in range(num_test_we_use_LFB):
        for j in range(args.seq):
            test_idx_LFB.append(test_we_use_start_idx_LFB[i] + j)

    dict_index, dict_value = zip(*list(enumerate(test_we_use_start_idx_LFB)))
    dict_test_start_idx_LFB = dict(zip(dict_value, dict_index))

    print('num test start idx : {:6d}'.format(len(test_useful_start_idx)))
    print('last idx test start: {:6d}'.format(test_useful_start_idx[-1]))
    print('num of test dataset: {:6d}'.format(num_test))
    print('num of test we use : {:6d}'.format(num_test_we_use))
    print('num of all test use: {:6d}'.format(num_test_all))

# TODO sampler

    test_loader = DataLoader(test_dataset,
                             batch_size=args.test,
                             sampler=SeqSampler(test_dataset, test_idx),
                             num_workers=args.work)

    global g_LFB_test
    print("loading features!>.........")

    # args.load_LFB = False
    if not args.load_LFB:

        test_feature_loader = DataLoader(
            test_dataset,
            batch_size=args.test,
            sampler=SeqSampler(test_dataset, test_idx_LFB),
            num_workers=args.work,
            pin_memory=False
        )

        model_LFB = resnet_lstm_LFB()        
        model_LFB.load_state_dict(torch.load(args.lfb_model), strict=False)

        if args.gpu:
            model_LFB = DataParallel(model_LFB)
            model_LFB.to(device)

        for params in model_LFB.parameters():
            params.requires_grad = False

        model_LFB.eval()

        with torch.no_grad():

            for data in test_feature_loader:
                if args.gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                inputs = inputs.view(-1, args.seq, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs)

                for j in range(len(outputs_feature)):
                    save_feature = outputs_feature.data.cpu()[j].numpy()
                    save_feature = save_feature.reshape(1, 512)
                    g_LFB_test = np.concatenate((g_LFB_test, save_feature), axis=0)

                print("train feature length:", len(g_LFB_test))

        print("finish!")
        g_LFB_test = np.array(g_LFB_test)

        with open("g_LFB_train_train.pkl", 'wb') as f:
            pickle.dump(g_LFB_test, f)

    else:
        with open("g_LFB_train_train.pkl", 'rb') as f:
            g_LFB_test = pickle.load(f)

        print("load completed")

    print("g_LFB_test shape:", g_LFB_test.shape)

    torch.cuda.empty_cache()

    model = resnet_lstm()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")  
    model.load_state_dict(torch.load(args.model))
    model = DataParallel(model)

    if args.gpu:
        model.to(device)
    # 应该可以直接多gpu计算
    # model = model.module            #要测试一下
    criterion = nn.CrossEntropyLoss(reduction='sum', weight=unique_class_weights.to(device))

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []
    all_preds_score = []


    y_true = []

    with torch.no_grad():

        df_probs = pd.DataFrame()
        for data in test_loader:
            
            # 释放显存
            torch.cuda.empty_cache()            

            if args.gpu:
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels[(args.seq - 1)::args.seq]
            else:
                inputs, labels = data
                labels = labels[(args.seq - 1)::args.seq]

            start_index_list = data[2]
            start_index_list = start_index_list[0::args.seq]
            long_feature = get_long_feature(start_index_list=start_index_list,
                                            dict_start_idx_LFB=dict_test_start_idx_LFB,
                                            lfb=g_LFB_test)
            
            long_feature = torch.Tensor(long_feature).to(device)

            inputs = inputs.view(-1, args.seq, 3, 224, 224)

            outputs = model.forward(inputs, long_feature=long_feature)

            # outputs = outputs[args.seq - 1::args.seq]
            Sm = nn.Softmax()
            outputs = Sm(outputs)

            possibility, preds = torch.max(outputs.data, 1)
            # print("possibility:",possibility)

            y_true.append(labels)
            for i in range(len(preds)):
                all_preds.append(preds[i].data.cpu())
            for i in range(len(possibility)):
                all_preds_score.append(possibility[i].data.cpu())
            # print("all_preds length:",len(all_preds))
            # print("all_preds_score length:",len(all_preds_score)) 
            loss = criterion(outputs, labels)
            # TODO 和batchsize相关
            # test_loss += loss.data[0]/test_loss += loss.data.item()
            # print("preds:",preds.data.cpu())
            # print("labels:",labels.data.cpu())

            test_loss += loss.data.item()
            test_corrects += torch.sum(preds == labels.data)
            # print("test_corrects:",test_corrects)

    test_elapsed_time = time.time() - test_start_time
    test_accuracy = float(test_corrects) / float(num_test_we_use)
    test_average_loss = test_loss / num_test_we_use
    

    y_true = torch.cat(y_true,dim=0).cpu()
    all_preds = np.array(all_preds)

    print(classification_report(y_true,all_preds))

    print('type of all_preds:', type(all_preds))
    print('leng of all preds:', len(all_preds))
    save_test = int("{:4.0f}".format(test_accuracy * 10000))
    pred_name = model_pure_name + '_test_' + str(save_test) + '_crop_' + str(args.crop) + '.pkl'
    pred_score_name = model_pure_name + '_test_' + str(save_test) + '_crop_' + str(args.crop) +'_score'+'.pkl'


    ## save output probabilty in csv or pickle 
    with open(pred_name, 'wb') as f:
        pickle.dump(all_preds, f)
    with open(pred_score_name, 'wb') as f:
        pickle.dump(all_preds_score, f)
    print('test elapsed: {:2.0f}m{:2.0f}s'
          ' test loss: {:4.4f}'
          ' test accu: {:.4f}'
          .format(test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_average_loss, test_accuracy))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lstm testing')
    parser.add_argument('--csv', type=str, help='input csv file')
    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--lfb_model', type=str, help='mempry bank model')

    parser.add_argument('-g', '--gpu', default=True, type=bool, help='use gpu, default True')
    parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
    parser.add_argument('-t', '--test', default=400, type=int, help='test batch size, default 10')
    parser.add_argument('-w', '--work', default=4, type=int, help='num of args.work to use, default 4')
    parser.add_argument(
        '-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 2 resize, 5 five_crop, 10 ten_crop, default 2')
    parser.add_argument('--LFB_l', default=30, type=int, help='long term feature bank length')
    parser.add_argument('--load_LFB', default=True, type=bool, help='whether load exist long term feature bank')


    args = parser.parse_args()
    print_params(args)
    test_model(args)

print('Done')
print()
