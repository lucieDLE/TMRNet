import os
import numpy as np
import pickle
from utils import get_dirs2, get_files2
import argparse

##
## script creating picke file splitting train/val/test dataset (contains paths and label)
##

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=True, type=str, help='data directory')
parser.add_argument('--cholec', default='.', type=str, help='output file path (.pkl)')
parser.add_argument('--test_label', default='.', type=str, help='output file path (.pkl)')

args = parser.parse_args()


## put in arguments
img_dir2 = os.path.join(args.data, 'frames')
phase_dir2 = os.path.join(args.data, 'phase_annotations')
tool_dir2 = os.path.join(args.data, 'tool_annotations')

print(args.data)
print(img_dir2)
print(phase_dir2)

#cholec80==================
img_dir_names2, img_dir_paths2 = get_dirs2(img_dir2)
tool_file_names2, tool_file_paths2 = get_files2(tool_dir2)
phase_file_names2, phase_file_paths2 = get_files2(phase_dir2)

phase_dict = {}
phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)
#cholec80==================


#cholec80==================
all_info_all2 = []

for j in range(len(img_dir_paths2)):
    downsample_rate = 25
    phase_file = open(phase_file_paths2[j])
    tool_file = open(tool_file_paths2[j])

    video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths2[j]))[0][5:7])
    video_num_dir = int(os.path.basename(img_dir_paths2[j]))

    print("video_num_file:", video_num_file,"video_num_dir:", video_num_dir, "rate:", downsample_rate)

    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split()
        if first_line:
            first_line = False
            continue
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths2[j], phase_split[0] + '.jpg')
            if os.path.exists(img_file_each_path):
                info_each.append(img_file_each_path)
                info_each.append(phase_dict[phase_split[1]])
                info_all.append(info_each)              

    # print(len(info_all))
    all_info_all2.append(info_all)
#cholec80==================

with open(args.outfile, 'wb') as f:
    pickle.dump(all_info_all2, f)

with open(args.outfile, 'rb') as f:
    all_info_80 = pickle.load(f)


#cholec80==================
train_file_paths_80 = []
test_file_paths_80 = []
val_file_paths_80 = []
val_labels_80 = []
train_labels_80 = []
test_labels_80 = []

train_num_each_80 = []
val_num_each_80 = []
test_num_each_80 = []

# print(all_info_80) # [video 1[[f1, label], ..., [fn, label]], video2[[f2,label]...[fn,label]], ...]

for i in range(40): ## train of 40 videos
    # print(all_info_80[i]) # access 1st video and so list of [fn, label]
    train_num_each_80.append(len(all_info_80[i])) # list going from frame 1 to frame n
    for j in range(len(all_info_80[i])):
        train_file_paths_80.append(all_info_80[i][j][0]) #take path of frame
        train_labels_80.append(all_info_80[i][j][1:])#take label of frame

print(len(train_file_paths_80))
print(len(train_labels_80))

for i in range(40,48):
    val_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        val_file_paths_80.append(all_info_80[i][j][0])
        val_labels_80.append(all_info_80[i][j][1:])

print(len(val_file_paths_80))
print(len(val_labels_80))

for i in range(40,80):
    test_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        test_file_paths_80.append(all_info_80[i][j][0])
        test_labels_80.append(all_info_80[i][j][1:])

print(len(test_file_paths_80))
print(len(test_labels_80))


#cholec80==================


train_val_test_paths_labels = []
train_val_test_paths_labels.append(test_file_paths_80)

train_val_test_paths_labels.append(test_labels_80)

train_val_test_paths_labels.append(test_num_each_80)

with open(args.test_label, 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)

print('Done')
print()
