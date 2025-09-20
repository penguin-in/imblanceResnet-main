#Plot the 3D results.
import numpy as np
import matplotlib.pyplot as plt
import re

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

filename = 'resneten.txt'
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

x = []
y = []
z = []

for i, line in enumerate(lines):
    if 'testAcc2:' in line and 'testAcc2num:' in line:
        testAcc2_match = re.search(r'testAcc2: (\d+\.?\d*)%', line)
        if testAcc2_match:
            x_val = float(testAcc2_match.group(1)) * 0.01
        else:
            x_val = np.nan

        testAcc2num_match = re.search(r'testAcc2num: (\d+\.?\d*)', line)
        if testAcc2num_match:
            y_val = float(testAcc2num_match.group(1))
        else:
            y_val = np.nan

        if not np.isnan(x_val) and not np.isnan(y_val) and x_val > 0.58:
            z_val = y_val * (x_val - 0.58) ** 1.25
            x.append(x_val)
            y.append(y_val)
            z.append(z_val)

x = np.array(x)
y = np.array(y)
z = np.array(z)

valid_indices = (x > 0.70) & (y > 24) & (z > 3)
x_filtered = x[valid_indices]
y_filtered = y[valid_indices]
z_filtered = z[valid_indices]

if len(x_filtered) == 0:
    x_filtered = x
    y_filtered = y
    z_filtered = z

fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

ax.set_facecolor('white')

ax.scatter(x_filtered, y_filtered, z_filtered, s=50, c='black', alpha=0.8)

ax.set_xlabel('Precision', fontsize=20)
ax.set_ylabel('TP', fontsize=20)
ax.set_zlabel('S', fontsize=20)
ax.set_title('EfficientNet', fontsize=24, pad=10)

ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.zaxis.set_major_locator(plt.MaxNLocator(3))

max_z_idx = np.argmax(z_filtered)
max_x_idx = np.argmax(x_filtered)
max_y_idx = np.argmax(y_filtered)

ax.scatter(x_filtered[max_z_idx], y_filtered[max_z_idx], z_filtered[max_z_idx],
           s=200, c='#FF9999', edgecolors='black', linewidth=1, alpha=0.8, label='Max Z')

ax.scatter(x_filtered[max_x_idx], y_filtered[max_x_idx], z_filtered[max_x_idx],
           s=200, c='#99FF99', edgecolors='black', linewidth=1, alpha=0.8, label='Max X')

ax.scatter(x_filtered[max_y_idx], y_filtered[max_y_idx], z_filtered[max_y_idx],
           s=200, c='#9999FF', edgecolors='black', linewidth=1, alpha=0.8, label='Max Y')

x_mesh = np.linspace(np.min(x_filtered), np.max(x_filtered), 20)
y_mesh = np.linspace(np.min(y_filtered), np.max(y_filtered), 20)
x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
z_plane = y_mesh * (x_mesh - 0.58) ** 1.25

surf = ax.plot_surface(x_mesh, y_mesh, z_plane, alpha=0.3, edgecolor='none', cmap='viridis')

ax.tick_params(axis='both', which='major', labelsize=16)

ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

