#Plot the results.
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams['font.size'] = 12  # 基础字体大小

filename = 'resnetvit.txt'
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

x = []
y = []

for i, line in enumerate(lines):
    if 'testAcc2:' in line and 'testAcc2num:' in line:
        testAcc2_match = re.search(r'testAcc2: (\d+\.?\d*)%', line)
        if testAcc2_match:
            y_val = float(testAcc2_match.group(1)) * 0.01
        else:
            y_val = np.nan

        testAcc2num_match = re.search(r'testAcc2num: (\d+\.?\d*)', line)
        if testAcc2num_match:
            x_val = float(testAcc2num_match.group(1))
        else:
            x_val = np.nan

        if not np.isnan(x_val) and not np.isnan(y_val):
            x.append(x_val)
            y.append(y_val)

x = np.array(x)
y = np.array(y)

plt.figure(figsize=(12, 8))
plt.scatter(x, y, s=80, c='black', alpha=0.7)
plt.xlabel('TP', fontsize=24)  #
plt.ylabel('Precision', fontsize=24)  #
plt.title('TP vs Precision (ViT)', fontsize=20)
plt.grid(True)

p = np.polyfit(x, y, 1)
y_fit = np.polyval(p, x)

plt.plot(x, y_fit, 'r-', linewidth=4)

equation_str = f'y = {p[0]:.4f}x + {p[1]:.4f}'
x_pos = np.min(x) + (np.max(x) - np.min(x)) * 0.1
y_pos = np.max(y) - (np.max(y) - np.min(y)) * 0.1
plt.text(x_pos, y_pos, equation_str, fontsize=20,
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.8'))

plt.legend(['Data Points', 'Fitted Line'], loc='best', fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.show()
