from comet_ml import Experiment
import io 
import matplotlib.pyplot as plt
import torch
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
from sklearn import metrics
import pandas as pd 

from dataset import *
from models import *
import pdb

from sklearn.utils import class_weight


def print_parameters(args):
    num_gpu = torch.cuda.device_count()

    print('number of gpu   : {:6d}'.format(num_gpu))
    print('sequence length : {:6d}'.format(args.num_frames))
    print('train batch size: {:6d}'.format(args.train_bs))
    print('valid batch size: {:6d}'.format(args.val_bs))
    print('optimizer choice: {:6d}'.format(args.opt))
    print('multiple optim  : {:6d}'.format(args.multi))
    print('num of epochs   : {:6d}'.format(args.epochs))
    print('num of workers  : {:6d}'.format(args.num_workers))
    print('test crop type  : {:6d}'.format(args.crop))
    print('whether to flip : {:6d}'.format(args.flip))
    print('learning rate   : {:.4f}'.format(args.lr))
    print('momentum for sgd: {:.4f}'.format(args.momentum))
    print('weight decay    : {:.4f}'.format(args.weight_decay))
    print('dampening       : {:.4f}'.format(args.dampening))
    print('use nesterov    : {:6d}'.format(args.nesterov))
    print('method for sgd  : {:6d}'.format(args.sgdadjust))
    print('step for sgd    : {:6d}'.format(args.sgdstep))
    print('gamma for sgd   : {:.4f}'.format(args.sgdgamma))


def get_csv_data(data_path, split='train'):
    df = pd.read_csv(data_path)

    paths = df['frame'].to_numpy()
    labels = df['class'].to_numpy()
    n_classes = np.unique(labels)
    list_videos_length = []

    for class_name in n_classes:
        df_tmp = df.loc[df['class']==class_name]
        len_vids = df_tmp['id'].value_counts().to_list()
        for elt in len_vids:
            list_videos_length.append(elt)

    print(len(list_videos_length))
    print('{}_paths  : {:6d}'.format(split, len(paths)))

    train_transforms = None
    test_transforms = None
    if args.flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])
    elif args.flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
        ])

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

    if split == 'train':
        dataset = CholecDataset(paths, labels, train_transforms)
    elif split=='val':
        dataset = CholecDataset(paths, labels, test_transforms)

    return dataset, list_videos_length

class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def display_sequence(video):
    batch_size =10
    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)

    video = video.data.cpu()
    print(video.shape)
    for idx in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, idx+1)
        img = video[idx,...]
        img = np.reshape(img, (224,224,3))
        plt.imshow(img)
    # plt.savefig(path)
    plt.close()
    return fig 

def get_list_sequences(list_frames_idx):
    list_sequences = []
    for frame_idx in list_frames_idx:
        for j in range(args.num_frames):
            list_sequences.append(frame_idx + j)

    return list_sequences


def get_start_idx_frames(num_frames, list_length_videos):
    count = 0
    list_frames_start_sequence = []
    for video_length in list_length_videos:
        ## for j in range 0 to of length video - 10 frames
        for j in range(count, count + (video_length + 1 - num_frames)):
            list_frames_start_sequence.append(j)
        count += video_length
    return list_frames_start_sequence


def main(args):
    # TensorBoard

    use_gpu = (torch.cuda.is_available() and args.gpu)
    device = torch.device("cuda:0" if use_gpu else "cpu")

    exp = Experiment(api_key='jvo9wdLqVzWla60yIWoCd0fX2',
                        project_name='TMRnet',
                        workspace='luciedle')
    
    train_ds, train_videos_length= get_csv_data(args.csv_train)
    val_ds, val_videos_length = get_csv_data(args.csv_valid)

    df_train = pd.read_csv(args.csv_train)

    unique_classes = np.sort(np.unique(df_train['class']))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train['class']))    
    unique_class_weights = torch.tensor(unique_class_weights).to(torch.float32)


    list_train_start_frame_idx = get_start_idx_frames(args.num_frames, train_videos_length)
    list_train_sequences = get_list_sequences(list_train_start_frame_idx)
    num_train_frames = len(list_train_start_frame_idx)

    list_val_start_frame_idx = get_start_idx_frames(args.num_frames, val_videos_length)
    list_val_sequences = get_list_sequences(list_val_start_frame_idx)
    val_loader = DataLoader(val_ds, batch_size=args.val_bs, sampler=SeqSampler(val_ds, list_val_sequences), num_workers=args.num_workers, pin_memory=False)



    pdb.set_trace()
    criterion_phase = nn.CrossEntropyLoss(reduction='sum', weight=unique_class_weights.to(device))
    model = resnet_lstm(args, num_class=unique_classes.shape[0])
       

    optimizer = None
    exp_lr_scheduler = None

    optimizer = model.get_optimizers()

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    if use_gpu:
        model = DataParallel(model)
        model.to(device)
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        np.random.shuffle(list_train_start_frame_idx)
        list_train_sequences =get_list_sequences(list_train_start_frame_idx)
        train_loader_80 = DataLoader(train_ds, batch_size=args.train_bs, sampler=SeqSampler(train_ds, list_train_sequences), num_workers=args.num_workers, pin_memory=False)

        # Sets the module in training mode.
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()

        for i, data in enumerate(train_loader_80):
            optimizer.zero_grad()

            inputs, labels_phase = data[0].to(device), data[1].to(device)

            labels_phase = labels_phase[(args.num_frames - 1)::args.num_frames]

            inputs = inputs.view(-1, args.num_frames, 3, 224, 224)
            outputs_phase = model.forward(inputs)
            outputs_phase = outputs_phase[args.num_frames - 1::args.num_frames]

            
            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss = criterion_phase(outputs_phase, labels_phase)

            running_loss_phase += loss.data.item()
            train_loss_phase += loss.data.item()

            loss.backward()
            optimizer.step()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase


            if i % 500 == 499:
                running_loss = running_loss_phase / (float(args.train_bs*50))
                running_acc = float(minibatch_correct_phase) / (float(args.train_bs)*50)

                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            if (i+1)*args.train_bs >= num_train_frames:               
                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0

            batch_progress += 1
            if batch_progress*args.train_bs >= num_train_frames:
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', num_train_frames, num_train_frames), end='\n')
            else:
                percent = round(batch_progress*args.train_bs / num_train_frames * 100, 2)
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress*args.train_bs, num_train_frames), end='\r')

        train_accuracy_phase = float(train_corrects_phase) / float(num_train_frames) * args.num_frames
        train_average_loss_phase = train_loss_phase / num_train_frames * args.num_frames

        # Sets the module in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in val_loader:

                inputs, labels_phase = data[0].to(device), data[1].to(device)
                labels_phase = labels_phase[(args.num_frames - 1)::args.num_frames]

                outputs_phase = model.forward(inputs)
                outputs_phase = outputs_phase[args.num_frames - 1::args.num_frames]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                # TODO

                all_preds.append(preds_phase)
                all_labels.append(labels_phase)

                val_progress += 1
                if val_progress*args.val_bs >= len(list_val_sequences):
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', len(list_val_sequences), len(list_val_sequences)), end='\n')
                else:
                    percent = round(val_progress*args.val_bs / len(list_val_sequences) * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*args.val_bs, len(list_val_sequences)), end='\r')

        val_accuracy_phase = float(val_corrects_phase) / float(len(list_val_start_frame_idx))
        val_average_loss_phase = val_loss_phase / len(list_val_start_frame_idx)

        val_recall_phase = metrics.recall_score(all_labels,all_preds, average='macro')
        val_precision_phase = metrics.precision_score(all_labels,all_preds, average='macro')

        
        print('epoch: {:4d} train loss: {:4.4f}  train accu: {:.4f}'
              ' valid loss: {:4.4f} valid accu: {:.4f}'
              .format(epoch, train_average_loss_phase, train_accuracy_phase,
                    val_average_loss_phase, val_accuracy_phase))

        print("val_precision_phase", val_precision_phase)
        print("val_recall_phase", val_recall_phase)

        train_metrics = {"training loss": running_loss, "traininc acc": running_acc,
                         "valid loss":val_average_loss_phase,"valid acc":val_accuracy_phase,
                         "val_precision": val_precision_phase, "val_recall": val_recall_phase,
        }
        exp.log_metrics(train_metrics, epoch=epoch)
        exp.log_confusion_matrix(all_labels, all_preds,epoch=epoch)


        if args.opt == 0:
            if args.sgdadjust == 0:
                exp_lr_scheduler.step()
            elif args.sgdadjust == 1:
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
                     + "_length_" + str(args.num_frames) \
                     + "_batch_" + str(args.train_bs) \
                     + "_train_" + str(save_train_phase) \
                     + "_val_" + str(save_val_phase)

        torch.save(best_model_wts, "./best_model/"+base_name+".pth")
        print("best_epoch",str(best_epoch))

        torch.save(model.module.state_dict(), "./temp/latest_model_"+str(epoch)+".pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lstm training')

    parser.add_argument('--csv_train', help='CSV for training', type=str, required=True)
    parser.add_argument('--csv_valid', help='CSV for validation', type=str, required=True)

    parser.add_argument('--num_frames', default=10, type=int, help='sequence length, default 10')

    parser.add_argument('--gpu', default=True, type=bool, help='gpu use, default True')
    parser.add_argument('--num_workers', default=8, type=int, help='num of workers to use, default 4')

    parser.add_argument('--epochs', default=25, type=int, help='epochs to train and val, default 25')
    parser.add_argument('--train_bs', default=100, type=int, help='train batch size, default 400')
    parser.add_argument('--val_bs', default=40, type=int, help='valid batch size, default 10')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
    parser.add_argument('--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
    parser.add_argument('--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')

    parser.add_argument('--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
    parser.add_argument('--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
    parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
    parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
    parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
    parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
    parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
    parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')

    args = parser.parse_args()

    main(args)

print('Done')
print()