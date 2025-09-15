import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.stats import wilcoxon
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon

def read_and_average_roc(parent_dir):
    fpr_values = []
    tpr_values = []
    
    for folder in sorted(os.listdir(parent_dir)):  
        folder_path = os.path.join(parent_dir, folder)
        
        fpr_path = os.path.join(folder_path, "fpr_trans.csv")
        tpr_path = os.path.join(folder_path, "tpr_trans.csv")
        
        if os.path.exists(fpr_path) and os.path.exists(tpr_path):
            fpr = pd.read_csv(fpr_path, header=None).values.flatten()
            tpr = pd.read_csv(tpr_path, header=None).values.flatten()
            
            # Interpolate TPR values to a common FPR scale
            common_fpr = np.linspace(0, 1, 100)  
            interp_tpr = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=0)(common_fpr)
            
            fpr_values.append(common_fpr)
            tpr_values.append(interp_tpr)
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tpr_values, axis=0) if tpr_values else np.zeros_like(mean_fpr)
    std_tpr = np.std(tpr_values, axis=0) if tpr_values else np.zeros_like(mean_fpr)

    # Compute AUC mean and std
    auc_values = [auc(mean_fpr, tpr) for tpr in tpr_values]
    mean_auc = np.mean(auc_values) if auc_values else 0
    std_auc = np.std(auc_values) if auc_values else 0

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc

def main():

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #### Bar plots
    subjects = [f"S{i:02d}" for i in range(1, 15)]
    width = 0.2
    x = np.arange(len(subjects))
    methods = ["EEGCubeNet", "3D-EEGNet"]


    # # Accuracy, Cross-subject, HC vs RE
    # accuracy_means = {
    #     'EEGCubeNet':[60.00, 84.16, 85.83, 60.00, 65.83, 60.83, 77.50, 65.83, 82.5, 73.33, 66.66, 73.33, 68.33, 67.5] ,
    #     'EEGNet3D': [63.33, 89.16, 66.66, 60.00, 66.66, 66.66, 65.83, 63.33, 80.00, 60.83, 58.33, 68.33, 70.83, 65.83]

    # }

    ### Accuracy hcre with 10-folds avg
    #accuracy_means = {
    #    'EEGCubeNet':[60.00, 84.16, 85.83, 60.00, 65.83, 60.83, 77.50, 65.83, 82.5, 73.33, 66.66, 73.33, 68.33, 67.5] ,
    #    'EEGNet3D': [65.00, 82.41, 74.25, 60.16, 64.41, 69.41, 62.33, 60.00, 74.00, 65.58, 56.41, 65.25, 63.41, 66.16]
    #}

    # # Accuracy, Cross-subject, HO vs RE
    # accuracy_means = {
    #     'EEGCubeNet':[58.33, 84.16, 80.00, 60.83, 65.83, 69.16, 70.00, 60.83, 80.00, 72.50, 55.00, 68.33, 70.00, 72.50] ,
    #     'EEGNet3D': [63.33, 80.83, 65.00, 63.33, 64.16, 73.33, 59.16, 65.00, 73.33, 68.33, 60.83, 74.16, 62.50, 61.66]

    # }
    # Accuracy, Cross-subject, HO vs RE 10'fold
    accuracy_means = {
        'EEGCubeNet':[58.33, 84.16, 80.00, 60.83, 65.83, 69.16, 70.00, 60.83, 80.00, 72.50, 55.00, 68.33, 70.00, 72.50] ,
        '3D-EEGNet':  [62.16, 76.83, 77.00, 64.58, 64.41, 69.33, 67.66, 59.16, 78.41, 69.00, 53.68, 67.00, 64.50, 67.08]
    }


    plt.figure(figsize=(8, 5))
    for i, method in enumerate(methods):
       plt.bar(x + i * width, accuracy_means[method], width=width, label=method) #yerr=accuracy_stds[method], capsize=3
    #     # plt.violinplot(accuracy_means[method],showmeans=True, showextrema=True, showmedians=True)

    plt.xlabel('Subjects')
    plt.ylabel('Cross-Subject Test Accuracy (%)')
    # plt.title('HCRE: Accuracy')
    plt.xticks(x + (len(methods) * width) / 2 - width / 2, subjects)
    plt.legend()
    plt.show()


    # # Boxplot
    
    # # Old boxplot with space, not suitable for double coulmn figure

    # subjects = [f"S{i:02d}" for i in range(1, 15)]

    # accuracy_hcre_P = [60.00, 84.16, 85.83, 60.00, 65.83, 60.83, 77.50, 65.83, 82.5, 73.33, 66.66, 73.33, 68.33, 67.5]
    # accuracy_hcre_E = [63.33, 89.16, 66.66, 60.00, 66.66, 66.66, 65.83, 63.33, 80.00, 60.83, 58.33, 68.33, 70.83, 65.83]
    # # Create DataFrame 
    # data1 = {
    #     "Subject": subjects * 2,
    #     "Fine-tune Accuracy": accuracy_hcre_P + accuracy_hcre_E,
    #     "Model": ["EEGCubeNet"] * 14 + ["EEGNet3D"] * 14
    # }
    # df1 = pd.DataFrame(data1)
    # #### wilcoxon test
    # # Perform Wilcoxon signed-rank test
    # stat, p_value = wilcoxon(accuracy_hcre_P, accuracy_hcre_E, alternative='greater')
    # # Plot
    # plt.figure(figsize=(6.5, 5))
    # ax = sns.boxplot(x="Model", y="Fine-tune Accuracy", data=df1, width=0.15, palette=["royalblue", "orangered"], orient="v", showfliers=False)
    # sns.stripplot(x="Model", y="Fine-tune Accuracy", data=df1, color="black", alpha=0.2, jitter=True)
    # # Add statistical annotation with a line
    # y_max = max(accuracy_hcre_P + accuracy_hcre_E) + 1.5  # Position above max value
    # x1, x2 = 0, 1  # x positions of the two boxplots
    # plt.plot([x1, x1, x2, x2], [y_max-0.2, y_max, y_max, y_max-0.2], lw=0.4, color='black')  # Horizontal line
    # plt.text((x1 + x2) / 2, y_max, f"p = {p_value:.3f}", ha='center', fontsize=10, fontweight='normal')
    # # Labels
    # plt.ylabel("Cross-Subject Test Accuracy(%)")
    # plt.grid(axis='y', linestyle='--', alpha=0.2)
    # plt.tight_layout()
    # # Show plot
    # plt.show()


    ##### Following latest results for the wilcosn plots and averaged over 10 folds.

    subjects = [f"S{i:02d}" for i in range(1, 15)]
    # hore normal latest
    accuracy_hcre_P = [60.00, 84.16, 85.83, 60.00, 65.83, 60.83, 77.50, 65.83, 82.5, 73.33, 66.66, 73.33, 68.33, 67.5] 
    # hore 10'fold lates
    # accuracy_hcre_P = [60.00, 84.16, 85.83, 60.00, 65.83, 60.83, 77.50, 65.83, 82.5, 73.33, 66.66, 73.33, 68.33, 67.5]
    ## with 10-fold avg hcre latest
    accuracy_hcre_E = [65.00, 82.41, 74.25, 60.16, 64.41, 69.41, 62.33, 60.00, 74.00, 65.58, 56.41, 65.25, 63.41, 66.16]
    ## with 10-fold avg hore latest
    # accuracy_hcre_E = [62.16, 76.83, 77.00, 64.58, 64.41, 69.33, 67.66, 59.16, 78.41, 69.00, 53.68, 67.00, 64.50, 67.08]

    # Wilcoxon test
    stat, p_value = wilcoxon(accuracy_hcre_P, accuracy_hcre_E, alternative='greater')

    # Plot
    plt.figure(figsize=(4, 5))
    positions = [0.9, 1.1]
    colors = ['royalblue', 'orangered']

    # Draw each boxplot manually with different colors and transparency
    for i, (data, color) in enumerate(zip([accuracy_hcre_P, accuracy_hcre_E], colors)):
        bp = plt.boxplot(data, positions=[positions[i]], widths=0.1, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.5),  # Semi-transparent
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black')
                        # flierprops=dict(marker='o', color='black', alpha=0.2)
                        )

        # Scatter dots (strip plot)
        jitter = (np.random.rand(len(data)) - 0.5) * 0.05
        plt.scatter([positions[i] + j for j in jitter], data, color='white', alpha=0.0, zorder=0)

    # Statistical annotation
    y_max = max(accuracy_hcre_P + accuracy_hcre_E) + 1.5
    plt.plot([positions[0], positions[0], positions[1], positions[1]],
            [y_max - 0.2, y_max, y_max, y_max - 0.2], lw=0.6, color='black')
    plt.text(np.mean(positions), y_max + 0.3, f"p = {p_value:.3f} (<0.05)", ha='center', fontsize=10.5)

    # Axes and labels
    plt.xticks(positions, ["EEGCubeNet", "3D-EEGNet"])
    plt.ylabel("Cross-Subject Test Accuracy (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.xlim(0.75, 1.25)
    plt.tight_layout()
    # plt.savefig('/home/sufflin/Documents/MotionPlanning/IJNS-2025/EEGNet/results/hore.png', dpi = 300)
    plt.show()


    #### Wilcoxon for trainng-performance for 10 fold accuracy

    
    ## HCRE 

    # Provided model accuracy data (converted to float arrays)
    model_1 = {
        "S01": ['76.60', '70.51', '86.86', '68.27', '79.17', '79.49', '84.62', '80.77', '79.49', '78.21'],
        "S02": ['56.09', '71.79', '68.27', '75.64', '82.05', '75.32', '84.29', '85.26', '85.58', '78.53'],
        "S03":  ['65.06', '81.41', '82.37', '78.85', '63.14', '70.19', '74.36', '74.04', '73.40', '82.37'],
        "S04":  ['63.14', '81.09', '75.00', '78.21', '88.14', '72.76', '90.71', '76.92', '76.60', '67.95'],
        "S05":  ['66.35', '82.37', '75.00', '81.73', '76.28', '74.68', '72.44', '74.04', '58.33', '84.29'],
        "S06":  ['84.94', '55.13', '83.65', '71.79', '84.62', '60.90', '85.58', '66.03', '61.22', '87.82'],
        "S07":  ['63.46', '70.51', '82.69', '71.79', '79.81', '87.82', '75.64', '77.88', '75.64', '75.64'],
        "S08":  ['51.60', '80.77', '85.90', '75.96', '70.83', '71.15', '72.76', '85.26', '76.60', '70.83'],
        "S09":  ['53.53', '52.56', '79.81', '76.60', '70.83', '75.00', '73.72', '66.35', '80.77', '67.63'],
        "S10":  ['62.82', '84.94', '84.62', '73.72', '82.05', '77.24', '76.28', '86.54', '73.40', '85.90'],
        "S11":  ['76.28', '80.77', '83.33', '81.41', '89.10', '87.82', '74.36', '83.33', '71.79', '89.42'],
        "S12": ['75.00', '81.73', '67.63', '53.85', '72.76', '78.85', '62.50', '85.26', '76.92', '68.27'],
        "S13": ['68.91', '85.26', '77.56', '84.94', '83.97', '78.21', '80.45', '72.44', '72.44', '84.94'],
        "S14":  ['59.62', '84.62', '75.96', '88.14', '76.60', '86.22', '63.46', '79.81', '73.72', '76.92']
        }
    
    model_2 = {
        "S01": [72.10, 73.14, 92.31, 74.68, 87.82, 86.54, 88.78, 90.17, 86.86, 80.13],
        "S02": [88.46, 89.42, 86.22, 89.42, 86.12, 88.20, 84.21, 70.90, 85.90, 91.99],
        "S03": [84.29, 84.29, 85.25, 87.40, 84.10, 81.09, 81.09, 83.54, 82.15, 88.14],
        "S04": [67.63, 88.78, 82.67, 82.37, 80.77, 91.03, 86.22, 81.09, 86.54, 91.35],  
        "S05": [77.56, 86.86, 87.50, 89.42, 79.49, 89.10, 80.77, 89.42, 83.01, 74.68],
        "S06": [83.01, 87.18, 76.60, 88.46, 87.82, 92.31, 70.83, 82.37, 86.22, 92.63],
        "S07": [91.99, 90.38, 78.85, 90.38, 84.94, 87.50, 84.94, 84.29, 89.74, 85.26],
        "S08":  [88.46, 90.38, 91.35, 86.86, 90.38, 90.06, 77.88, 87.50, 82.05, 88.78],
        "S09": [73.40, 70.19, 78.85, 80.77, 90.06, 90.06, 89.74, 86.54, 78.21, 90.38],
        "S10":  [89.42, 89.10, 74.68, 81.09, 88.46, 90.38, 89.10, 86.22, 83.01, 83.65],
        "S11": [79.17, 88.78, 83.65, 88.14, 76.28, 94.23, 90.38, 82.05, 92.31, 92.63],
        "S12":[91.35, 85.26, 80.45, 87.82, 91.03, 64.42, 90.06, 90.38, 92.95, 89.10],
        "S13": [83.65, 75.32, 82.37, 88.78, 88.14, 81.09, 87.18, 82.37, 83.65, 85.26],
        "S14":  [88.46, 85.26, 76.60, 86.86, 89.10, 91.03, 89.10, 85.26, 91.03, 94.87]
    }
   
    """
    ## HORE
    model_1 = {
        "S01": ['77.24', '54.49', '78.53', '70.51', '75.00', '84.29', '77.24', '72.76', '70.83', '79.17'],
        "S02": ['82.69', '65.38', '71.15', '87.82', '64.74', '82.05', '59.94', '84.29', '76.60', '74.36'],
        "S03": ['72.44', '78.21', '58.65', '80.13', '80.45', '69.87', '86.54', '83.33', '77.24', '91.99'],
        "S04": ['86.54', '57.05', '77.88', '72.44', '72.44', '66.99', '57.05', '86.54', '81.09', '72.44'],
        "S05": ['70.83', '77.88', '69.55', '79.81', '77.56', '78.21', '91.67', '79.81', '70.83', '92.95'], 
        "S06": ['85.58', '78.21', '72.76', '77.88', '83.01', '74.68', '80.77', '80.13', '63.78', '87.50'],
        "S07": ['79.81', '63.14', '71.47', '75.00', '62.50', '90.06', '75.96', '72.44', '71.47', '80.77'],
        "S08": ['77.88', '75.64', '74.36', '92.31', '81.73', '68.59', '75.96', '92.95', '86.86', '85.58'],
        "S09": ['84.94', '71.47', '66.03', '63.78', '66.67', '80.13', '75.96', '75.64', '77.56', '87.82'],
        "S10": ['75.96', '56.41', '53.21', '63.78', '77.56', '87.50', '81.73', '91.35', '73.72', '87.18'],
        "S11": ['58.33', '80.77', '79.81', '72.76', '78.53', '76.60', '84.94', '60.90', '81.73', '93.27'],
        "S12": ['68.59', '61.22', '87.50', '78.85', '78.53', '73.72', '55.13', '75.32', '65.06', '71.47'],
        "S13": ['87.18', '75.32', '77.24', '83.65', '78.21', '67.63', '75.96', '86.86', '81.09', '65.71'],
        "S14": ['80.13', '83.65', '84.29', '72.44', '80.13', '90.06', '76.92', '73.40', '83.33', '77.88']
    }

    model_2 = {
        "S01":  [89.74, 78.21, 87.18, 91.35, 91.67, 91.67, 86.54, 90.06, 90.06, 80.77],
        "S02":  [93.59, 72.12, 83.33, 90.71, 88.46, 90.71, 77.88, 81.09, 90.38, 68.91],
        "S03":  [87.50, 83.33, 91.99, 81.73, 89.74, 89.10, 87.50, 71.79, 89.10, 81.73],
        "S04":  [89.10, 91.03, 78.85, 83.97, 94.23, 90.71, 69.55, 85.26, 92.63, 86.86],
        "S05":  [89.74, 88.46, 78.53, 88.46, 89.10, 85.58, 87.18, 90.71, 91.03, 88.46],
        "S06": [86.22, 75.96, 85.90, 85.58, 82.05, 85.90, 88.14, 85.26, 85.58, 90.38],
        "S07":  [80.45, 88.46, 88.14, 85.26, 88.14, 88.46, 89.74, 92.31, 87.82, 88.78],
        "S08":  [79.81, 90.71, 88.46, 89.42, 87.50, 85.90, 90.06, 89.10, 82.37, 91.67],
        "S09": [91.67, 88.14, 91.03, 85.26, 91.67, 87.82, 73.08, 84.62, 91.67, 89.10],
        "S10": [89.10, 91.99, 88.14, 85.26, 89.42, 85.90, 86.86, 80.45, 84.94, 89.10],
        "S11": [79.49, 84.94, 79.81, 81.41, 92.63, 88.78, 89.10, 85.90, 87.50, 84.29],
        "S12": [90.06, 79.17, 91.67, 91.67, 89.42, 90.38, 90.71, 89.74, 90.71, 82.05],
        "S13":  [82.69, 86.54, 77.24, 89.74, 82.05, 87.82, 83.33, 92.31, 91.99, 92.63],
        "S14": [87.50, 73.40, 81.41, 75.96, 91.99, 87.18, 76.60, 87.50, 88.78, 70.19]
    }
    """


    # Flatten and convert string to float
    acc_model_1 = np.array([float(acc) for accs in model_1.values() for acc in accs])
    acc_model_2 = np.array([acc for accs in model_2.values() for acc in accs])

    # Wilcoxon Signed-Rank Test (paired)
    stat, p_value = wilcoxon(acc_model_1, acc_model_2)

    # Print result
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value:.5f}")

    # Plot
    plt.figure(figsize=(4, 5))
    positions = [0.9, 1.1]
    colors = ['royalblue', 'orangered']

    # Draw each boxplot manually with different colors and transparency
    for i, (data, color) in enumerate(zip([acc_model_1, acc_model_2], colors)):
        bp = plt.boxplot(data, positions=[positions[i]], widths=0.1, patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.5),  # Semi-transparent
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black')
                        # flierprops=dict(marker='o', color='black', alpha=0.2)
                        )

        # Scatter dots (strip plot)
        jitter = (np.random.rand(len(data)) - 0.5) * 0.05
        plt.scatter([positions[i] + j for j in jitter], data, color='white', alpha=0.0, zorder=0.0)

    # Statistical annotation
    # y_max = max(acc_model_1 + acc_model_2) + 0.0
    y_max = max(np.max(acc_model_1), np.max(acc_model_2)) + 0.5
    plt.plot([positions[0], positions[0], positions[1], positions[1]],
            [y_max - 0.2, y_max, y_max, y_max - 0.2], lw=0.6, color='black')
    plt.text(np.mean(positions), y_max + 0.2, f"p = {p_value:.3f} (<0.05)", ha='center', fontsize=10.5)

    # Axes and labels
    
    plt.xticks(positions, ["EEGCubeNet", "3D-EEGNet"])
    plt.ylabel("Cross-Subject Train Accuracy (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.xlim(0.75, 1.25)
    plt.ylim(0, y_max + 5.0)  # Enough margin for annotation line and text
    plt.tight_layout()
    plt.savefig('/home/sufflin/Documents/MotionPlanning/IJNS-2025/EEGNet/results/hcre_train.png', dpi = 300)
    plt.show()




    ### scatterplot with stat. significance diagonal line

    # # Data (Replace these with actual accuracy values)
    # subjects = [f"S{i+1}" for i in range(14)]  # Subject label

    # # Perform Wilcoxon signed-rank test
    # stat, p_value = wilcoxon(accuracy_hcre_P, accuracy_hcre_E, alternative='greater')

    # # Create DataFrame
    # df1 = pd.DataFrame({
    #     "Subject": subjects,
    #     "EEGCubeNet": accuracy_hcre_P,
    #     "3D-EEGNet": accuracy_hcre_E
    # })

    # # Plot scatter points with subject markers
    # plt.figure(figsize=(8, 8))
    # plt.scatter(df1["EEGCubeNet"], df1["3D-EEGNet"], marker='+', s=90, label="Subjects")

    # # Diagonal Line (Reference for Statistical Significance)
    # x_vals = np.linspace(min(df1["EEGCubeNet"].min(), df1["3D-EEGNet"].min()),
    #                      max(df1["EEGCubeNet"].max(), df1["3D-EEGNet"].max()), 100)
    # plt.plot(x_vals, x_vals, linestyle='--', color='black', label=f"Wilcoxon p={p_value:.4f} (<0.05)")

    # # Labels and Formatting
    # plt.xlabel("EEGCubeNet Accuracy (%)")
    # plt.ylabel("3D-EEGNet Accuracy (%)")
    # # plt.title("Model Accuracy Comparison per Subject")
    # plt.legend()
    # plt.grid(True)

    # # Show plot
    # plt.show()


if __name__ == "__main__":
    main()