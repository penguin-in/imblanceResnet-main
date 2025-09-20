# This code is used for the weight distribution analysis of seeds at different germination stages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm

file_path = 'datasheet.xlsx'
all_sheets = pd.read_excel(file_path, sheet_name=None)

ori_data = np.concatenate((all_sheets['2'], all_sheets['3'], all_sheets['4'],
                           all_sheets['5'], all_sheets['6'], all_sheets['7'], all_sheets['8'],
                           all_sheets['9'], all_sheets['10'], all_sheets['11']), axis=0)

raw_data = ori_data[:, 6]
raw_data = np.array(raw_data, dtype=np.float64)
raw_data = raw_data[~np.isnan(raw_data)]

mu_fit, sigma_fit = norm.fit(raw_data)
print(f"Normal Distribution Parameters: μ={mu_fit:.4f}, σ={sigma_fit:.4f}")
print(f"Data Range: {raw_data.min():.2f} to {raw_data.max():.2f}")
print(f"Sample Size: {len(raw_data):,}")

plt.figure(figsize=(16, 12))
plt.rcParams['font.family'] = 'DejaVu Sans'

n, bins, patches = plt.hist(raw_data, bins=64, density=True, alpha=0.8,
                           label="Data Histogram", edgecolor='white', linewidth=0.5,
                           color='#2E86AB', zorder=2)

x = np.linspace(raw_data.min(), raw_data.max(), 1000)
pdf_norm = norm.pdf(x, loc=mu_fit, scale=sigma_fit)
plt.plot(x, pdf_norm, 'r-', lw=4, label=f"Normal Distribution Fit (μ={mu_fit:.2f}, σ={sigma_fit:.2f})",
         zorder=3, alpha=0.9)

plt.title('Seed day3 Weight Distribution',
          fontsize=26, fontweight='bold', pad=25, color='#2B2D42')
plt.xlabel('Weight', fontsize=22, fontweight='bold', labelpad=15, color='#2B2D42')
plt.ylabel('Probability Density', fontsize=22, fontweight='bold', labelpad=15, color='#2B2D42')

plt.xlim(raw_data.min() * 0.95, raw_data.max() * 1.05)

plt.legend(fontsize=18, frameon=True, fancybox=True, shadow=True,
           loc='upper right', framealpha=0.9)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True, alpha=0.2, zorder=1, linestyle='--', linewidth=0.5)

text_str = f'Parameters:\nμ = {mu_fit:.4f}\nσ = {sigma_fit:.4f}\nSample Size: {len(raw_data):,}\nRange: {raw_data.min():.2f}-{raw_data.max():.2f}'
plt.text(0.75, 0.75, text_str, transform=plt.gca().transAxes, fontsize=18,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#F8F9FA',
                                          alpha=0.9, edgecolor='#6C757D', linewidth=1),
         color='#2B2D42', linespacing=1.5)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

plt.tight_layout()

plt.savefig("seed_weight_distribution_raw.png",
            dpi=600,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

plt.show()