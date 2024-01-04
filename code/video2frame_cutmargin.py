# -----------------------------
# Cut black margin for surgical video
# Copyright (c) CUHK 2021. 
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------


import cv2
import os
import numpy as np
import PIL
from PIL import Image
import imageio

import pandas as pd 
source_path = "/CMF/data/lumargot/cholec80/videos/"  # original path
save_path = "/CMF/data/lumargot/cholec80/frames/"  # save path


def change_size(image):
 
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10,y-10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)
    
    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left  
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom  

    pre1_picture = image[left:left + width, bottom:bottom + height]  
    
    return pre1_picture  

def main():

    in_dir = '/CMF/data/jprieto/hysterectomy/data/Clips/'
    out_dir = '/CMF/data/lumargot/hysterectomy/'

    # for fold in os.listdir(in_dir):
    #     sub_fold = os.path.join(in_dir, fold)
    #     print(fold)
    #     if(os.path.isdir(sub_fold)):
    #         out_sub_dir = os.path.join(out_dir, fold)
    #         if not os.path.exists(out_sub_dir):
    #             os.mkdir(out_sub_dir)

    #         for file in os.listdir(sub_fold):
    #             vid_path  = os.path.join(sub_fold, file)

    #             splitted = file.split('_')
    #             name = ('_').join(splitted[:-2])
    #             save_path = os.path.join(out_sub_dir, name)
    #             ## save as name_frame_idx.png

    #             # Use imageio for video reading
    #             video_reader = imageio.get_reader(vid_path)
    #             frame_num = 0
    #             for frame in video_reader:
    #                 # Frame is a NumPy array, can be passed to OpenCV directly
    #                 if frame_num %25 ==0:
    #                     img_save_path = save_path + '_' + str(frame_num) + ".png"
    #                     img_result = PIL.Image.fromarray(frame)
    #                     img_result.save(img_save_path)

    #                 frame_num = frame_num + 1



    n_classes = []
    n_files = []
    n_id = []

    df  = pd.read_csv('../../scripts/hyst_ds_train_test_wo_label.csv')

    for fold in os.listdir(out_dir):
        sub_fold = os.path.join(out_dir, fold)
        print(fold)
        if(os.path.isdir(sub_fold)):

            for file in os.listdir(sub_fold):
                vid_path  = os.path.join(sub_fold, file)
                splitted = file.split('_')
                name = ('_').join(splitted[:-1])
                id = splitted[-2]
                if (df['vid_path'].str.contains(name).any()):
                    n_files.append(vid_path)
                    n_classes.append(fold)
                    n_id.append(id)
            
        
        
    df = pd.DataFrame(data={"frames":n_files,"class":n_classes, 'id':n_id})
    df.to_csv('../../scripts/val_frames.csv')

    
    print("Cut Done")

if __name__ == "__main__":
    main()