import os
import cv2
import numpy as np
from function import extract_color_region
import pandas as pd

import re
import shutil

image_dir = "/media/ls/办公/ls/seeddata/seeddata/prosessed_imag"
file_path = '/media/ls/办公/ls/seeddata/seeddata/sorted_output.xlsx'
save_path = '/media/ls/办公/ls/seeddata/seeddata/processed_purple'
save_path1 = '/media/ls/办公/ls/seeddata/seeddata/head_image'
save_path2 = '/media/ls/办公/ls/seeddata/seeddata/tail_image'
save_path3 = '/media/ls/办公/ls/seeddata/seeddata/bw_image'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
if os.path.exists(save_path1):
    shutil.rmtree(save_path1)
os.makedirs(save_path1)
if os.path.exists(save_path2):
    shutil.rmtree(save_path2)
os.makedirs(save_path2)
if os.path.exists(save_path3):
    shutil.rmtree(save_path3)
os.makedirs(save_path3)
all_sheets = pd.read_excel(file_path,sheet_name=None,header=None)
ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']
                    ,all_sheets['9'],all_sheets['10'],all_sheets['11']),axis = 0)
data = ori_data[:, 1:12]
vigor = data[:,10]
p_label = data[:,8]
lens = len(p_label)
predict_label = np.ones(lens)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".bmp"))]
image_files.sort(key=lambda x: int(re.search(r'\d+',x).group()))

if len(image_files) != len(vigor):
    raise ValueError(f"image_files_len:{len(image_files)}labels_len{len(vigor)}")
vigor_sum = 0
for idx,filename in enumerate(image_files):
    image_path = os.path.join(image_dir,filename)
    mask_image,head_image,tail_image,_,bw_image = extract_color_region(image_path=image_path)
    if np.all(mask_image == 0):
        predict_label[idx] = 0
    elif vigor[idx] == 1:
            vigor_sum += 1
    output_path = os.path.join(save_path, filename)
    output_path1 = os.path.join(save_path1, filename)
    output_path2 = os.path.join(save_path2, filename)
    output_path3 = os.path.join(save_path3, filename)
    cv2.imwrite(output_path3, bw_image)
    cv2.imwrite(output_path, mask_image)
    cv2.imwrite(output_path1, head_image)
    cv2.imwrite(output_path2, tail_image)

print(f"predict label accuracy: {vigor_sum*100/sum(predict_label):.3f}")
save_data = np.concatenate((data,predict_label.reshape(-1, 1)), axis=1)
df = pd.DataFrame(save_data)
df.to_excel("save_outputv6.xlsx", index=False, header=False)