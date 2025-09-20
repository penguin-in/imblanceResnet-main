#lower_hsv_destroy=(0, 0, 87),(255, 60, 255)，threshold=26
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
max_pva = 0
max_pla = 0
for t in range(1, 31 ,5):
    for i in range(60, 80, 2):
        for j in range(60, 90, 3):
            vigor_sum = 0
            predict_label = np.ones(lens)

            for idx, filename in enumerate(image_files):
                image_path = os.path.join(image_dir, filename)
                _, _, _,mask_image = extract_color_region(
                    image_path=image_path,
                    lower_hsv=(0, 0, 70),
                    upper_hsv=(20, 255, 255),
                    lower_hsv_destroy=(0, 0, j),
                    upper_hsv_destroy=(255, i, 255),
                    resize_shape=(1000, 1000),
                    kernel_size=(5, 5),
                    bw_threshold=40,
                    show=False
                )
                if np.count_nonzero(mask_image) < t:
                    predict_label[idx] = 0
                else:
                    predict_label[idx] = 1
                    if vigor[idx] == 0:
                        vigor_sum += 1

            if sum(predict_label) == 0:
                pva = 0
            else:
                pva = vigor_sum * 100 / sum(predict_label)

            pla = accuracy(predict_label, p_label) * 100

            if pva > max_pva:
                max_pva = pva
                pva_information = (pva, pla, i, j, t, sum(predict_label))

            if pla > max_pla:
                max_pla = pla
                pla_information = (pva, pla, i, j, t,sum(predict_label))
            print(
                f"epoch accuracy: PVA={pva:.3f}%, PLA={pla:.3f}%, HSV upperH={i}, lowerV={j},threshold={t}，num={sum(predict_label)}")


print(f"best predict vigor accuracy: PVA={pva_information[0]:.3f}%, PLA={pva_information[1]:.3f}%, HSV upperH={pva_information[2]}, lowerV={pva_information[3]}, threshold={pva_information[4]},num={pva_information[5]}")
print(f"best predict label accuracy: PVA={pla_information[0]:.3f}%, PLA={pla_information[1]:.3f}%, HSV upperH={pla_information[2]}, lowerV={pla_information[3]},,threshold={pla_information[4]},num={pla_information[5]}")
# save_data = np.concatenate((data,predict_label.reshape(-1, 1)), axis=1)
# df = pd.DataFrame(save_data)
# df.to_excel("/home/ls/code/seed/neural_network/save_output_v3.xlsx", index=False, header=False)