import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,  #
    'axes.labelsize': 24,
    'xtick.labelsize': 21,
    'ytick.labelsize': 21,
    'legend.fontsize': 21,
})

excel_file_path = 'sorted_output.xlsx'
data_path = 'entire_data.xlsx'

data0 = pd.read_excel(data_path, sheet_name='Sheet1')
data0 = np.array(data0, dtype=np.float64)
all_sheets = pd.read_excel(excel_file_path, sheet_name=None, header=None)
ori_data = np.concatenate((all_sheets['2'], all_sheets['3'], all_sheets['4'],
                           all_sheets['5'], all_sheets['6'], all_sheets['7'],
                           all_sheets['8'], all_sheets['9'], all_sheets['10'],
                           all_sheets['11']), axis=0)
data = np.concatenate((ori_data[:, 1:8], ori_data[:, 11].reshape(-1, 1)), axis=1)
data = np.array(data, dtype=np.float64)

data_vitality = data[:, 7]
weight_rate = data[:, 0]
weight = data[:, 2]
area = data0[:, 8]
primeter = data0[:, 9]


def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))


def find_optimal_threshold(fpr, tpr, thresholds):
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    return thresholds[optimal_idx], youden_index[optimal_idx]


def analyze_association(weight_data, vitality_data):
    vitality_data_modified = vitality_data.copy()

    vitality_data_modified[weight_data < 0.5] = 0

    df = pd.DataFrame({
        'weight': weight_data,
        'vitality': vitality_data_modified
    })

    # Basic statistics
    stats_summary = {
        'weight_stats': df['weight'].describe(),
        'vitality_distribution': df['vitality'].value_counts()
    }

    plt.figure(figsize=(12, 9))

    sns.histplot(data=df, x='weight', hue='vitality', element='step', common_norm=False)
    plt.title('SE Distribution Overlay', fontsize=24)
    plt.ylabel('Number of Seeds', fontsize=24)
    plt.xlabel('SE', fontsize=24)

    plt.tight_layout()
    plt.savefig("weight_vitality_distribution", dpi=2000)
    plt.show()

    point_biserial_corr, p_value = stats.pointbiserialr(df['vitality'], df['weight'])

    fpr, tpr, thresholds = roc_curve(df['vitality'], df['weight'])
    roc_auc = auc(fpr, tpr)

    optimal_threshold, youden_index = find_optimal_threshold(fpr, tpr, thresholds)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.scatter(fpr[np.argmax(youden_index)], tpr[np.argmax(youden_index)],
                color='red', s=100, label=f'Optimal threshold ({optimal_threshold:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.show()

    # 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        df['weight'].values.reshape(-1, 1),
        df['vitality'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['vitality'].values  # 保持类别比例
    )


    x0 = optimal_threshold
    k = 10.0  #

    y_pred_proba = sigmoid(X_test.flatten(), x0, k)
    y_pred = (y_pred_proba > 0.5).astype(int)


    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 8))

    x_range = np.linspace(df['weight'].min(), df['weight'].max(), 100)
    y_sigmoid = sigmoid(x_range, x0, k)

    plt.scatter(X_test[y_test == 0], y_test[y_test == 0], alpha=0.6,
                label='Actual Class 0', color='blue')
    plt.scatter(X_test[y_test == 1], y_test[y_test == 1], alpha=0.6,
                label='Actual Class 1', color='red')
    plt.scatter(X_test, y_pred, alpha=0.3, marker='x',
                label='Predicted', color='green')

    plt.plot(x_range, y_sigmoid, 'k-', lw=2, label='Sigmoid Function')
    plt.axvline(x=optimal_threshold, color='gray', linestyle='--',
                label=f'Threshold ({optimal_threshold:.3f})')
    plt.axhline(y=0.5, color='gray', linestyle=':')

    plt.xlabel('Seed Weight Growth Rate (SE)')
    plt.ylabel('Probability / Class')
    plt.title('Sigmoid Classification Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sigmoid_classification.png", dpi=300)
    plt.show()

    group0 = df[df['vitality'] == 0]['weight']
    group1 = df[df['vitality'] == 1]['weight']

    norm_test = {
        'group0': stats.shapiro(group0),
        'group1': stats.shapiro(group1)
    }

    if norm_test['group0'].pvalue > 0.05 and norm_test['group1'].pvalue > 0.05:
        test_result = stats.ttest_ind(group0, group1)
        test_used = 't-test'
    else:
        test_result = stats.mannwhitneyu(group0, group1)
        test_used = 'Mann-Whitney U'

    point_biserial = stats.pointbiserialr(df['vitality'], df['weight'])

    return {
        'summary_stats': stats_summary,
        'normality_tests': {
            'group0': {'W': norm_test['group0'].statistic, 'p': norm_test['group0'].pvalue},
            'group1': {'W': norm_test['group1'].statistic, 'p': norm_test['group1'].pvalue}
        },
        'group_comparison': {
            'test': test_used,
            'statistic': test_result.statistic,
            'p_value': test_result.pvalue
        },
        'correlation': {
            'point_biserial': point_biserial.correlation,
            'p_value': point_biserial.pvalue
        },
        'advanced_metrics': {
            'correlation_coefficient': point_biserial_corr,
            'auc_score': roc_auc,
            'optimal_threshold': optimal_threshold,
            'test_accuracy': accuracy,
            'confusion_matrix': cm.tolist()
        }
    }


if __name__ == "__main__":
    results = analyze_association(weight_rate, data_vitality)

