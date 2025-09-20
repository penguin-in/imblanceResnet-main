#循环寻找参数，最佳参数upperH=20,lowerV=70
import os
import cv2
import numpy as np
from function import extract_color_region
import pandas as pd
from function import accuracy
import re
import shutil

image_dir = "/media/ls/办公/ls/seeddata/seeddata/prosessed_imag"
file_path = '/media/ls/办公/ls/seeddata/seeddata/sorted_output.xlsx'
save_path = '/media/ls/办公/ls/seeddata/seeddata/processed_purple'
save_path1 = '/media/ls/办公/ls/seeddata/seeddata/head_image'
save_path2 = '/media/ls/办公/ls/seeddata/seeddata/tail_image'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
if os.path.exists(save_path1):
    shutil.rmtree(save_path1)
os.makedirs(save_path1)
if os.path.exists(save_path2):
    shutil.rmtree(save_path2)
os.makedirs(save_path2)

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
max_pva = 0
max_pla = 0
for i in range(15, 26):
    for j in range(40, 80):
        vigor_sum = 0
        predict_label = np.ones(lens)

        for idx, filename in enumerate(image_files):
            image_path = os.path.join(image_dir, filename)
            mask_image, head_image, tail_image = extract_color_region(
                image_path=image_path,
                lower_hsv=(0, 0, j),
                upper_hsv=(i, 255, 255),
                resize_shape=(1000, 1000),
                kernel_size=(5, 5),
                bw_threshold=40,
                show=False
            )
            if np.all(mask_image == 0):
                predict_label[idx] = 0
            else:
                predict_label[idx] = 1
                if vigor[idx] == 1:
                    vigor_sum += 1

        if sum(predict_label) == 0:
            pva = 0
        else:
            pva = vigor_sum * 100 / sum(predict_label)

        pla = accuracy(predict_label, p_label) * 100

        if pva > max_pva:
            max_pva = pva
            pva_imformation = (pva, pla, i, j, sum(predict_label))

        if pla > max_pla:
            max_pla = pla
            pla_information = (pva, pla, i, j, sum(predict_label))
        print(f"epoch accuracy: PVA={pva:.3f}%, PLA={pla:.3f}%, HSV upperH={i}, lowerV={j},num={sum(predict_label)}")

print(f"best predict vigor accuracy: PVA={pva_imformation[0]:.3f}%, PLA={pva_imformation[1]:.3f}%, HSV upperH={pva_imformation[2]}, lowerV={pva_imformation[3]},num={pva_imformation[4]}")
print(f"best predict label accuracy: PVA={pla_information[0]:.3f}%, PLA={pla_information[1]:.3f}%, HSV upperH={pla_information[2]}, lowerV={pla_information[3]},num={pva_imformation[4]}")
# save_data = np.concatenate((data,predict_label.reshape(-1, 1)), axis=1)
# df = pd.DataFrame(save_data)
# df.to_excel("/home/ls/code/seed/neural_network/save_output_v3.xlsx", index=False, header=False)