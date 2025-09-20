import os
import cv2
import numpy as np
from function import extract_color_region
from function import accuracy
import pandas as pd

import re
import shutil

image_dir = "/media/ls/办公/ls/seeddata/seeddata/prosessed_imag"
file_path = '/media/ls/办公/ls/seeddata/seeddata/sorted_output.xlsx'
save_path = '/media/ls/办公/ls/seeddata/seeddata/destroy_image'

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

all_sheets = pd.read_excel(file_path,sheet_name=None,header=None)
ori_data = np.concatenate((all_sheets['2'],all_sheets['3'],all_sheets['4'],
                       all_sheets['5'],all_sheets['6'],all_sheets['7'],all_sheets['8']
                    ,all_sheets['9'],all_sheets['10'],all_sheets['11']),axis = 0)
data = ori_data[:, 1:12]
vigor = data[:,10]
p_label = data[:,7]
lens = len(p_label)
predict_label = np.ones(lens)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg",".png",".bmp"))]
image_files.sort(key=lambda x: int(re.search(r'\d+',x).group()))

if len(image_files) != len(vigor):
    raise ValueError(f"image_files_len:{len(image_files)}labels_len{len(vigor)}")
vigor_sum = 0
for idx,filename in enumerate(image_files):
    image_path = os.path.join(image_dir,filename)
    _,_,_,mask_image = extract_color_region(image_path=image_path,)
    if np.count_nonzero(mask_image) < 26:
        predict_label[idx] = 0
    elif vigor[idx] == 1:
            vigor_sum += 1
    output_path = os.path.join(save_path, filename)
    cv2.imwrite(output_path, mask_image)

print(f"predict label accuracy: {vigor_sum*100/sum(predict_label):.3f}")
print(f"predict vigor accuracy: {accuracy(predict_label,p_label)*100:.3f}")
save_data = np.concatenate((data,predict_label.reshape(-1, 1)), axis=1)
df = pd.DataFrame(save_data)
df.to_excel("/home/ls/code/seed/neural_network/save_outputv5.xlsx", index=False, header=False)