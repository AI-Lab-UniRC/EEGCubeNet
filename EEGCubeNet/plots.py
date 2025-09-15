import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from scipy.interpolate import interp1d
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.stats import wilcoxon
import matplotlib.patches as mpatches



def main():


    # Define the subjects
    subjects = [f"S{str(i).zfill(2)}" for i in range(1, 15)]

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches

    # Define example data (Replace with actual data)
    subjects = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    # Define boxplot positions
    positions = np.arange(len(subjects))  # Base x-axis positions
    width = 0.12  # Width spacing between boxplots

    convAll_fc = {
        "S01": ['0.83', '0.78', '0.83', '0.94', '0.83', '0.67', '0.78', '0.72', '0.89', '0.83'],
        "S02": ['0.94', '0.94', '0.89', '0.94', '0.94', '0.83', '0.89', '0.94', '0.94', '0.94'],
        "S03":  ['0.94', '0.72', '0.56', '0.89', '0.94', '0.78', '1.00', '0.94', '0.89', '0.94'],
        "S04":  ['0.83', '0.78', '0.83', '0.83', '0.72', '0.72', '0.61', '0.72', '0.78', '0.78'],
        "S05":  ['0.72', '0.50', '0.56', '0.83', '0.44', '0.72', '0.72', '0.78', '0.67', '0.61'],
        "S06":  ['0.67', '0.61', '0.67', '0.78', '0.72', '0.56', '0.67', '0.67', '0.56', '0.89'],
        "S07":  ['0.78', '0.94', '0.89', '0.94', '0.89', '0.94', '1.00', '0.67', '0.78', '0.83'],
        "S08":  ['0.28', '0.11', '0.06', '0.33', '0.72', '0.56', '0.22', '0.11', '0.22', '0.50'],
        "S09":  ['0.83', '0.89', '0.89', '0.89', '0.61', '0.67', '0.83', '0.83', '0.78', '0.83'],
        "S10":  ['0.89', '0.89', '0.94', '0.61', '0.94', '0.89', '0.83', '0.89', '1.00', '0.78'],
        "S11":  ['0.78', '0.72', '0.78', '0.89', '0.61', '0.78', '0.67', '0.83', '0.72', '0.67'],
        "S12":  ['0.67', '0.83', '0.67', '0.89', '0.78', '0.78', '0.67', '0.83', '0.83', '0.78'],
        "S13":  ['0.83', '0.89', '0.78', '0.72', '0.89', '0.89', '0.61', '0.78', '0.56', '0.83'], 
        "S14":  ['0.72', '0.83', '0.67', '0.61', '0.72', '0.56', '0.72', '0.61', '0.78', '0.78'],
    }
    convAll_fc = {key: [float(value) for value in values] for key, values in convAll_fc.items()}

    conv234_fc = {
        "S01": ['0.94', '0.67', '0.83', '0.94', '0.83', '0.67', '0.78', '0.83', '0.78', '0.89'] ,
        "S02": ['0.94', '0.94', '0.89', '0.89', '1.00', '0.89', '0.89', '0.78', '0.94', '0.89'],
        "S03":  ['0.83', '0.83', '1.00', '0.89', '0.94', '0.83', '0.94', '0.72', '0.94', '0.94'],
        "S04":  ['0.78', '0.78', '0.78', '0.72', '0.72', '0.72', '0.72', '0.78', '0.67', '0.78'],
        "S05":  ['0.61', '0.56', '0.50', '0.56', '0.50', '0.44', '0.61', '0.67', '0.72', '0.67'],
        "S06":  ['0.72', '0.78', '0.72', '0.89', '0.50', '0.56', '0.72', '0.78', '0.67', '0.83'],
        "S07":  ['0.83', '0.83', '0.89', '0.61', '0.72', '0.94', '1.00', '0.67', '0.78', '0.72'],
        "S08":  ['0.78', '0.22', '0.83', '0.33', '0.39', '0.28', '0.44', '0.06', '0.61', '0.28'],
        "S09":  ['0.89', '0.89', '0.83', '0.89', '0.78', '0.72', '0.89', '0.83', '0.89', '0.83'],
        "S10":  ['0.94', '0.83', '0.94', '0.83', '0.89', '0.83', '0.94', '0.89', '0.94', '0.94'],
        "S11":  ['0.83', '0.83', '0.78', '0.89', '0.61', '0.78', '0.83', '0.72', '0.67', '0.67'],
        "S12":  ['0.50', '0.61', '0.89', '0.83', '0.67', '0.78', '0.56', '0.61', '0.89', '0.56'],
        "S13":  ['0.83', '0.83', '0.56', '0.67', '0.44', '0.83', '0.83', '0.89', '0.67', '0.56'], 
        "S14":  ['0.78', '0.78', '0.78', '0.78', '0.72', '0.56', '0.67', '0.61', '0.67', '0.78'],
    }
    conv234_fc = {key: [float(value) for value in values] for key, values in conv234_fc.items()}


    conv34_fc = {
        "S01": ['0.94', '0.89', '0.89', '0.83', '0.72', '0.72', '0.78', '0.83', '0.83', '0.89'],
        "S02": ['1.00', '0.94', '0.83', '0.94', '1.00', '0.89', '1.00', '1.00', '1.00', '0.94'],
        "S03": ['0.94', '0.94', '0.94', '0.94', '0.89', '0.83', '0.89', '0.94', '0.94', '0.89'],
        "S04": ['0.78', '0.83', '0.89', '0.72', '0.89', '0.72', '0.72', '0.72', '0.72', '0.78'],
        "S05": ['0.56', '0.61', '0.44', '0.72', '0.78', '0.33', '0.56', '0.61', '0.67', '0.50'],
        "S06": ['0.78', '0.72', '0.72', '0.78', '0.67', '0.78', '0.83', '0.83', '0.89', '0.67'],
        "S07": ['0.61', '0.83', '0.78', '0.83', '0.61', '0.72', '0.78', '0.83', '0.78', '0.89'],
        "S08": ['0.17', '0.28', '0.83', '0.44', '0.72', '0.56', '0.33', '0.56', '0.61', '0.72'],
        "S09": ['0.89', '0.89', '0.94', '0.83', '0.89', '0.83', '0.83', '0.83', '0.78', '0.83'],
        "S10": ['0.94', '0.94', '0.78', '0.94', '0.89', '0.89', '0.83', '0.94', '0.89', '0.83'],
        "S11": ['0.78', '0.56', '0.78', '0.56', '0.61', '0.44', '0.72', '0.72', '0.72', '0.61'],
        "S12": ['0.67', '0.83', '0.72', '0.61', '0.72', '0.61', '0.67', '0.61', '0.67', '0.67'],
        "S13": ['0.78', '0.94', '0.83', '0.89', '0.89', '0.67', '0.89', '0.83', '0.83', '0.72'],
        "S14": ['0.72', '0.67', '0.78', '0.67', '0.78', '0.67', '0.44', '0.67', '0.72', '0.78'],
    }
    conv34_fc = {key: [float(value) for value in values] for key, values in conv34_fc.items()}


    # Conv_FC
    conv_fc = {
        "S01": ['0.78', '0.78', '0.72', '0.61', '0.72', '0.56', '0.67', '0.83', '0.83', '0.67'],
        "S02": ['0.78', '0.78', '0.83', '0.78', '0.72', '0.78', '0.78', '0.78', '0.72', '0.94'],
        "S03": ['0.78', '0.83', '0.94', '0.89', '0.72', '0.94', '0.94', '0.83', '0.94', '0.94'],
        "S04": ['0.78', '0.72', '0.50', '0.72', '0.72', '0.67', '0.67', '0.78', '0.78', '0.72'],
        "S05": ['0.56', '0.72', '0.44', '0.28', '0.50', '0.33', '0.50', '0.72', '0.61', '0.50'],
        "S06": ['0.78', '0.83', '0.61', '0.78', '0.72', '0.78', '0.83', '0.72', '0.83', '0.83'],
        "S07": ['0.83', '0.72', '0.72', '0.89', '0.89', '0.61', '0.50', '0.61', '0.89', '0.89'],
        "S08": ['0.33', '0.28', '0.28', '0.56', '0.28', '0.22', '0.22', '0.33', '0.17', '0.44'],
        "S09": ['0.83', '0.83', '0.78', '0.78', '0.89', '0.83', '0.89', '0.83', '0.83', '0.89'],
        "S10": ['0.83', '0.94', '0.83', '0.89', '0.94', '0.89', '0.94', '0.89', '0.89', '0.89'],
        "S11": ['0.33', '0.39', '0.22', '0.61', '0.39', '0.17', '0.50', '0.50', '0.33', '0.61'],
        "S12": ['0.39', '0.67', '0.67', '0.44', '0.56', '0.44', '0.44', '0.61', '0.56', '0.56'],
        "S13": ['0.67', '0.67', '0.67', '0.72', '0.72', '0.83', '0.78', '0.83', '0.78', '0.83'],
        "S14": ['0.44', '0.56', '0.67', '0.56', '0.61', '0.67', '0.56', '0.39', '0.56', '0.50'],
    }
    conv_fc = {key: [float(value) for value in values] for key, values in conv_fc.items()}

    # FC1 to FC3 data
    fc1 = {
        "S01": ['0.39', '0.39', '0.50', '0.61', '0.50', '0.44', '0.33', '0.50', '0.50', '0.56'],
        "S02": ['0.78', '0.67', '0.72', '0.67', '0.72', '0.72', '0.72', '0.67', '0.67', '0.78'],
        "S03": ['0.67', '0.67', '0.83', '0.72', '0.72', '0.67', '0.72', '0.72', '0.61', '0.78'],
        "S04": ['0.28', '0.72', '0.61', '0.61', '0.44', '0.67', '0.33', '0.56', '0.78', '0.28'],
        "S05": ['0.39', '0.28', '0.50', '0.44', '0.39', '0.39', '0.44', '0.44', '0.44', '0.39'],
        "S06": ['0.72', '0.67', '0.67', '0.61', '0.83', '0.72', '0.67', '0.72', '0.72', '0.61'],
        "S07": ['0.50', '0.56', '0.56', '0.50', '0.50', '0.50', '0.50', '0.39', '0.44', '0.50'],
        "S08": ['0.17', '0.06', '0.22', '0.17', '0.28', '0.06', '0.33', '0.22', '0.22', '0.22'],
        "S09": ['0.78', '0.78', '0.78', '0.72', '0.72', '0.83', '0.78', '0.83', '0.83', '0.89'],
        "S10": ['0.83', '0.94', '0.78', '0.72', '0.72', '0.94', '0.83', '0.83', '0.67', '0.83'],
        "S11": ['0.39', '0.33', '0.22', '0.11', '0.28', '0.39', '0.22', '0.22', '0.22', '0.22'],
        "S12": ['0.33', '0.28', '0.33', '0.22', '0.39', '0.44', '0.22', '0.17', '0.33', '0.44'],
        "S13": ['0.61', '0.56', '0.67', '0.61', '0.67', '0.67', '0.44', '0.72', '0.61', '0.61'],
        "S14": ['0.44', '0.67', '0.39', '0.50', '0.50', '0.50', '0.44', '0.44', '0.61', '0.50'],

    }
    fc1 = {key: [float(value) for value in values] for key, values in fc1.items()}
    
    fc2 = {
        "S01": ['0.44', '0.39', '0.28', '0.50', '0.50', '0.44', '0.33', '0.44', '0.44', '0.33'],
        "S02": ['0.72', '0.78', '0.72', '0.67', '0.67', '0.72', '0.67', '0.72', '0.61', '0.83'],
        "S03": ['0.67', '0.72', '0.78', '0.67', '0.56', '0.78', '0.78', '0.72', '0.61', '0.72'],
        "S04": ['0.39', '0.28', '0.28', '0.28', '0.33', '0.33', '0.28', '0.28', '0.28', '0.28'],
        "S05": ['0.33', '0.28', '0.56', '0.33', '0.67', '0.50', '0.28', '0.28', '0.28', '0.22'],
        "S06": ['0.61', '0.50', '0.72', '0.67', '0.61', '0.61', '0.78', '0.72', '0.67', '0.72'],
        "S07": ['0.50', '0.50', '0.50', '0.50', '0.44', '0.50', '0.56', '0.56', '0.50', '0.56'],
        "S08": ['0.44', '0.17', '0.22', '0.11', '0.17', '0.00', '0.39', '0.11', '0.17', '0.33'],
        "S09": ['0.72', '0.72', '0.72', '0.83', '0.83', '0.72', '0.83', '0.78', '0.67', '0.72'],
        "S10": ['0.72', '0.67', '0.78', '0.72', '0.78', '0.83', '0.61', '0.72', '0.67', '0.61'],
        "S11": ['0.28', '0.33', '0.33', '0.39', '0.17', '0.17', '0.11', '0.22', '0.11', '0.22'],
        "S12": ['0.33', '0.28', '0.28', '0.33', '0.28', '0.33', '0.33', '0.33', '0.11', '0.28'],
        "S13": ['0.56', '0.56', '0.50', '0.50', '0.61', '0.50', '0.33', '0.61', '0.56', '0.44'],
        "S14": ['0.39', '0.44', '0.44', '0.50', '0.44', '0.50', '0.44', '0.39', '0.61', '0.56'],
    }
    fc2 = {key: [float(value) for value in values] for key, values in fc2.items()}

    fc3 = {
        "S01": ['0.39', '0.39', '0.44', '0.39', '0.44', '0.39', '0.39', '0.39', '0.44', '0.39'],
        "S02": ['0.56', '0.78', '0.61', '0.61', '0.67', '0.61', '0.67', '0.61', '0.61', '0.72'],
        "S03": ['0.61', '0.67', '0.78', '0.67', '0.50', '0.72', '0.72', '0.61', '0.61', '0.72'],
        "S04": ['0.22', '0.33', '0.33', '0.39', '0.39', '0.39', '0.28', '0.39', '0.33', '0.44'],
        "S05": ['0.28', '0.39', '0.22', '0.44', '0.39', '0.44', '0.22', '0.39', '0.28', '0.50'],
        "S06": ['0.67', '0.61', '0.61', '0.61', '0.61', '0.56', '0.67', '0.72', '0.67', '0.67'],
        "S07": ['0.50', '0.50', '0.50', '0.50', '0.44', '0.50', '0.44', '0.56', '0.50', '0.50'],
        "S08": ['0.50', '0.22', '0.33', '0.11', '0.28', '0.00', '0.11', '0.17', '0.39', '0.22'],
        "S09": ['0.78', '0.72', '0.78', '0.72', '0.78', '0.83', '0.72', '0.72', '0.83', '0.72'],
        "S10": ['0.67', '0.72', '0.61', '0.72', '0.61', '0.78', '0.72', '0.67', '0.72', '0.67'],
        "S11": ['0.11', '0.39', '0.33', '0.28', '0.44', '0.06', '0.17', '0.06', '0.17', '0.11'],
        "S12": ['0.33', '0.28', '0.22', '0.33', '0.28', '0.22', '0.33', '0.28', '0.28', '0.17'],
        "S13": ['0.50', '0.39', '0.50', '0.39', '0.56', '0.44', '0.56', '0.44', '0.56', '0.39'],
        "S14": ['0.50', '0.56', '0.44', '0.44', '0.61', '0.56', '0.61', '0.44', '0.50', '0.56'],
    }
    fc3 = {key: [float(value) for value in values] for key, values in fc3.items()}


    # HCRE boxplots kappa kfold

    fc1_data = [fc1[subj] for subj in subjects]
    fc2_data = [fc2[subj] for subj in subjects]
    fc3_data = [fc3[subj] for subj in subjects]
    conv_fc_data = [conv_fc[subj] for subj in subjects]
    # conv_bn_fc_data = [conv_bn_fc[subj] for subj in subjects]
    conv34_fc_data = [conv34_fc[subj] for subj in subjects]
    conv234_fc_data = [conv234_fc[subj] for subj in subjects]
    convAll_fc_data = [convAll_fc[subj] for subj in subjects]

    # Create boxplot
    plt.figure(figsize=(20, 8))
    positions = np.arange(len(subjects))

    # Plot each dataset with an offset
    plt.boxplot(fc3_data, positions=positions, widths=width, patch_artist=True, boxprops=dict(facecolor="dimgray"), medianprops=dict(color="black"))
    plt.boxplot(fc2_data, positions=positions + width, widths=width, patch_artist=True, boxprops=dict(facecolor="brown"), medianprops=dict(color="black"))
    plt.boxplot(fc1_data, positions=positions + 2 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="olive"), medianprops=dict(color="black"))
    plt.boxplot(conv_fc_data, positions=positions + 3 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="lightblue"), medianprops=dict(color="black"))
    plt.boxplot(conv34_fc_data, positions=positions + 4 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="purple"), medianprops=dict(color="black"))
    plt.boxplot(conv234_fc_data, positions=positions + 5 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="blue"), medianprops=dict(color="black"))
    plt.boxplot(convAll_fc_data, positions=positions + 6 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="gold"), medianprops=dict(color="black"))
    
    # Create legend manually
    fc3_patch = mpatches.Patch(color='dimgray', label='FC3')
    fc2_patch = mpatches.Patch(color='brown', label='FC2+')
    fc1_patch = mpatches.Patch(color='olive', label='FC1+')
    conv_fc_patch = mpatches.Patch(color='lightblue', label='Conv4+')
    conv34_fc_patch = mpatches.Patch(color='purple', label='Conv3+')
    conv234_fc_patch = mpatches.Patch(color='blue', label='Conv2+')
    convAll_fc_patch = mpatches.Patch(color='gold', label='All-layers')

    
    ### for horizontal line to show mean
    # Compute the mean accuracy across all subjects for both FC-Finetuning and Conv-FC-Finetuning
    fc3_means = [sum(values) / len(values) for values in fc3_data]
    fc2_means = [sum(values) / len(values) for values in fc2_data]
    fc1_means = [sum(values) / len(values) for values in fc1_data]
    conv_fc_means = [sum(values) / len(values) for values in conv_fc_data]
    conv34_fc_means = [sum(values) / len(values) for values in conv34_fc_data]
    conv234_fc_means = [sum(values) / len(values) for values in conv234_fc_data]
    convAll_fc_means = [sum(values) / len(values) for values in convAll_fc_data]

    overall_fc3_mean = sum(fc3_means) / len(fc3_means)
    overall_fc2_mean = sum(fc2_means) / len(fc2_means)
    overall_fc1_mean = sum(fc1_means) / len(fc1_means)
    overall_conv_fc_mean = sum(conv_fc_means) / len(conv_fc_means)
    overall_conv34_fc_mean = sum(conv34_fc_means) / len(conv34_fc_means)
    overall_conv234_fc_mean = sum(conv234_fc_means) / len(conv234_fc_means)
    overall_convAll_fc_mean = sum(convAll_fc_means) / len(convAll_fc_means)

    # Plot horizontal average lines
    plt.axhline(overall_fc3_mean, color='dimgray', linestyle='--', linewidth=1, label='FC3+ Avg')
    plt.axhline(overall_fc2_mean, color='brown', linestyle='--', linewidth=1, label='FC2+ Avg')
    plt.axhline(overall_fc1_mean, color='olive', linestyle='--', linewidth=1, label='FC1+ Avg')
    plt.axhline(overall_conv_fc_mean, color='lightblue', linestyle='--', linewidth=1, label='Conv4+ Avg')
    plt.axhline(overall_conv34_fc_mean, color='purple', linestyle='--', linewidth=1, label='Conv3+ Avg')
    plt.axhline(overall_conv234_fc_mean, color='blue', linestyle='--', linewidth=1, label='Conv3+ Avg')
    plt.axhline(overall_convAll_fc_mean, color='gold', linestyle='--', linewidth=1, label='All-layers Avg')

    # X-axis settings
    plt.xticks(positions + (3 * width), subjects)  # Adjust labels to center
    plt.xlabel("Subjects")
    plt.ylabel("K-Cohen")
    plt.legend(handles=[fc3_patch, fc2_patch, fc1_patch, conv_fc_patch, conv34_fc_patch, conv234_fc_patch, convAll_fc_patch])
    plt.grid(False)
    plt.show()

    # HORE boxplots kappa kfold

    convAll_fc = {
        "S01":  ['0.61', '0.72', '0.78', '0.56', '0.56', '0.44', '0.50', '0.72', '0.67', '0.56'],
        "S02":  ['0.89', '0.78', '0.89', '0.89', '0.78', '0.67', '0.78', '0.67', '0.83', '0.83'],
        "S03":  ['0.83', '0.89', '0.83', '0.89', '0.89', '0.89', '0.83', '0.89', '0.83', '0.89'],
        "S04":  ['0.89', '0.94', '1.00', '0.94', '0.94', '0.89', '0.89', '0.94', '0.83', '0.83'],
        "S05":  ['0.83', '0.83', '0.61', '0.83', '0.72', '0.61', '0.78', '0.78', '0.78', '0.94'],
        "S06":  ['0.78', '0.78', '0.72', '0.67', '0.72', '0.61', '0.83', '0.83', '0.56', '0.83'],
        "S07":  ['0.89', '0.67', '0.89', '1.00', '0.89', '0.89', '0.83', '0.83', '0.94', '0.89'],
        "S08":  ['0.89', '0.56', '0.89', '0.72', '0.72', '0.89', '0.89', '0.61', '0.72', '0.83'],
        "S09":  ['1.00', '1.00', '0.89', '0.89', '0.83', '0.83', '1.00', '0.94', '0.83', '0.94'],
        "S10":  ['0.78', '0.83', '0.83', '0.72', '0.83', '0.83', '0.89', '0.83', '0.78', '0.83'],
        "S11":  ['0.61', '0.44', '0.61', '0.67', '0.67', '0.72', '0.67', '0.72', '0.50', '0.61'],
        "S12":  ['0.83', '0.50', '0.56', '0.83', '0.83', '0.56', '0.78', '0.67', '0.72', '0.50'],
        "S13":  ['0.28', '0.33', '0.28', '0.56', '0.83', '0.72', '0.50', '0.94', '0.22', '0.78'], 
        "S14":  ['0.78', '0.67', '0.83', '0.67', '0.83', '0.61', '0.78', '0.83', '0.89', '0.89'],
    }
    convAll_fc = {key: [float(value) for value in values] for key, values in convAll_fc.items()}

    conv234_fc = {
        "S01":  ['0.50', '0.67', '0.50', '0.78', '0.44', '0.44', '0.50', '0.61', '0.56', '0.50'],
        "S02":  ['0.89', '0.83', '0.89', '0.83', '0.78', '0.67', '0.78', '0.78', '0.89', '0.83'],
        "S03":  ['0.83', '0.94', '0.89', '0.94', '0.83', '0.89', '0.89', '0.89', '0.78', '0.89'],
        "S04":  ['0.89', '0.94', '1.00', '0.94', '1.00', '0.94', '0.83', '0.94', '0.89', '0.94'],
        "S05":  ['0.83', '0.78', '0.56', '0.78', '0.67', '0.67', '0.83', '0.89', '0.50', '0.94'],
        "S06":  ['0.33', '0.94', '0.83', '0.89', '0.89', '0.72', '0.83', '0.89', '0.94', '0.89'],
        "S07":  ['0.72', '0.83', '0.94', '0.78', '0.83', '0.89', '0.89', '0.94', '0.83', '0.78'],
        "S08":  ['0.83', '0.56', '0.94', '0.72', '0.61', '0.89', '0.89', '0.67', '0.72', '0.83'],
        "S09":  ['1.00', '1.00', '0.89', '0.89', '0.89', '0.89', '1.00', '0.89', '0.78', '0.89'],
        "S10":  ['0.78', '0.78', '0.72', '0.83', '0.83', '0.83', '0.89', '0.89', '0.83', '0.72'],
        "S11":  ['0.61', '0.44', '0.56', '0.61', '0.72', '0.67', '0.61', '0.61', '0.67', '0.50'],
        "S12":  ['0.83', '0.83', '0.83', '0.67', '0.72', '0.61', '0.61', '0.67', '0.56', '0.78'],
        "S13":  ['0.33', '0.67', '0.50', '0.78', '0.78', '0.72', '0.33', '0.78', '0.89', '0.78'], 
        "S14":  ['0.67', '0.83', '0.61', '0.61', '0.61', '0.56', '0.67', '0.83', '0.83', '0.78'],
    }
    conv234_fc = {key: [float(value) for value in values] for key, values in conv234_fc.items()}


    conv34_fc = {
        "S01": ['0.44', '0.67', '0.50', '0.61', '0.61', '0.56', '0.50', '0.50', '0.72', '0.67'],
        "S02": ['0.83', '0.83', '0.89', '0.83', '0.78', '0.67', '0.67', '0.89', '0.78', '0.78'],
        "S03": ['0.78', '0.94', '0.89', '0.89', '0.94', '0.89', '0.94', '0.89', '0.94', '0.94'],
        "S04": ['0.94', '0.94', '0.94', '0.94', '0.94', '0.89', '0.83', '0.94', '0.78', '0.94'],
        "S05": ['0.83', '0.78', '0.39', '0.78', '0.50', '0.44', '0.72', '0.67', '0.72', '0.67'],
        "S06": ['0.72', '0.94', '0.61', '0.72', '0.89', '0.78', '0.72', '0.89', '0.83', '0.78'],
        "S07": ['0.78', '0.94', '1.00', '0.83', '0.83', '0.83', '0.83', '0.83', '0.89', '0.94'],
        "S08": ['0.67', '0.72', '0.78', '0.78', '0.78', '0.94', '0.83', '0.78', '0.78', '0.61'],
        "S09": ['1.00', '0.94', '0.94', '1.00', '0.94', '0.83', '0.94', '0.83', '0.89', '0.89'],
        "S10": ['0.78', '0.83', '0.83', '0.78', '0.89', '0.83', '0.89', '0.89', '0.89', '0.89'],
        "S11": ['0.61', '0.67', '0.33', '0.67', '0.44', '0.50', '0.39', '0.61', '0.56', '0.44'],
        "S12": ['0.67', '0.78', '0.72', '0.78', '0.72', '0.56', '0.72', '0.72', '0.61', '0.78'],
        "S13": ['0.67', '0.50', '0.44', '0.50', '0.83', '0.61', '0.61', '0.67', '0.56', '0.61'],
        "S14": ['0.56', '0.83', '0.78', '0.56', '0.89', '0.56', '0.67', '0.67', '0.78', '0.78'],
    }
    conv34_fc = {key: [float(value) for value in values] for key, values in conv34_fc.items()}

    conv_fc = {
        "S01": ['0.61', '0.44', '0.50', '0.61', '0.61', '0.50', '0.56', '0.33', '0.61', '0.67'],
        "S02": ['0.72', '0.78', '0.78', '0.78', '0.78', '0.78', '0.67', '0.78', '0.72', '0.72'],
        "S03": ['0.89', '0.83', '0.83', '0.89', '0.83', '0.89', '0.83', '0.89', '0.89', '0.89'],
        "S04": ['0.33', '0.94', '0.33', '0.33', '0.94', '0.83', '0.67', '0.56', '0.17', '0.22'],
        "S05": ['0.89', '0.67', '0.72', '0.61', '0.22', '0.61', '0.33', '0.89', '0.39', '0.28'],
        "S06": ['0.56', '1.00', '0.61', '0.67', '0.61', '0.94', '0.78', '0.78', '0.67', '0.83'],
        "S07": ['0.83', '0.67', '0.78', '0.72', '0.89', '0.83', '0.78', '0.83', '0.89', '0.94'],
        "S08": ['0.39', '0.89', '0.72', '0.44', '0.83', '0.72', '0.50', '0.72', '0.72', '0.78'],
        "S09": ['0.89', '1.00', '0.78', '0.83', '0.94', '1.00', '0.89', '0.94', '1.00', '0.83'],
        "S10": ['0.89', '0.83', '0.89', '0.78', '0.83', '0.89', '0.83', '0.83', '0.83', '0.83'],
        "S11": ['0.28', '0.78', '0.33', '0.50', '0.22', '0.28', '0.28', '0.44', '0.44', '0.22'],
        "S12": ['0.56', '0.56', '0.61', '0.61', '0.67', '0.56', '0.56', '0.44', '0.72', '0.72'],
        "S13":['0.33', '0.50', '0.50', '0.67', '0.44', '0.39', '0.39', '0.61', '0.44', '0.67'],
        "S14": ['0.50', '0.44', '0.56', '0.61', '0.50', '0.67', '0.61', '0.83', '0.67', '0.44'],
    }
    conv_fc = {key: [float(value) for value in values] for key, values in conv_fc.items()}

    fc1 = {
        "S01": ['0.44', '0.50', '0.44', '0.50', '0.50', '0.39', '0.39', '0.39', '0.39', '0.33'],
        "S02": ['0.61', '0.67', '0.61', '0.67', '0.56', '0.61', '0.67', '0.56', '0.56', '0.67'],
        "S03": ['0.67', '0.72', '0.61', '0.83', '0.72', '0.72', '0.78', '0.83', '0.83', '0.89'],
        "S04": ['0.33', '0.06', '-0.17', '0.06', '0.06', '0.28', '-0.06', '0.22', '-0.06', '0.06'],
        "S05": ['0.33', '0.44', '0.33', '0.33', '0.28', '0.28', '0.22', '0.22', '0.17', '0.22'],
        "S06": ['0.44', '0.44', '0.50', '0.50', '0.56', '0.44', '0.56', '0.39', '0.56', '0.61'],
        "S07": ['0.72', '0.56', '0.50', '0.61', '0.44', '0.61', '0.56', '0.50', '0.28', '0.50'],
        "S08": ['0.44', '0.72', '0.56', '0.56', '0.61', '0.67', '0.44', '0.44', '0.78', '0.61'],
        "S09": ['0.78', '0.83', '0.78', '0.78', '0.67', '0.61', '0.61', '0.83', '0.83', '0.56'],
        "S10": ['0.72', '0.78', '0.72', '0.83', '0.72', '0.83', '0.78', '0.78', '0.83', '0.83'],
        "S11": ['0.00', '0.17', '-0.06', '0.33', '0.22', '0.22', '0.22', '0.11', '-0.06', '0.22'],
        "S12": ['0.44', '0.44', '0.50', '0.50', '0.56', '0.56', '0.56', '0.44', '0.50', '0.50'],
        "S13": ['0.33', '0.39', '0.44', '0.39', '0.50', '0.44', '0.33', '0.44', '0.33', '0.50'],
        "S14": ['0.33', '0.28', '0.44', '0.44', '0.44', '0.33', '0.33', '0.33', '0.44', '0.39'],
        
    }
    fc1 = {key: [float(value) for value in values] for key, values in fc1.items()}

    fc2 = {
    "S01": ['0.39', '0.50', '0.28', '0.44', '0.39', '0.39', '0.33', '0.39', '0.33', '0.28'],
    "S02": ['0.78', '0.67', '0.67', '0.67', '0.61', '0.67', '0.61', '0.72', '0.67', '0.61'],
    "S03": ['0.50', '0.72', '0.72', '0.67', '0.67', '0.67', '0.56', '0.72', '0.56', '0.61'],
    "S04": ['0.17', '0.00', '-0.11', '0.06', '-0.22', '0.00', '-0.17', '-0.06', '-0.33', '-0.06'],
    "S05": ['0.28', '0.28', '0.22', '0.17', '0.33', '0.33', '0.33', '0.22', '0.17', '0.11'],
    "S06": ['0.56', '0.61', '0.67', '0.56', '0.61', '0.39', '0.50', '0.44', '0.50', '0.61'],
    "S07": ['0.56', '0.56', '0.44', '0.44', '0.50', '0.50', '0.39', '0.39', '0.44', '0.44'],
    "S08": ['0.50', '0.61', '0.50', '0.61', '0.67', '0.67', '0.39', '0.33', '0.56', '0.44'],
    "S09": ['0.61', '0.56', '0.72', '0.61', '0.61', '0.61', '0.67', '0.56', '0.61', '0.61'],
    "S10": ['0.67', '0.67', '0.72', '0.67', '0.72', '0.67', '0.67', '0.72', '0.72', '0.72'],
    "S11": ['0.00', '0.28', '0.11', '0.28', '0.28', '-0.11', '0.06', '0.17', '0.11', '0.17'],
    "S12": ['0.56', '0.50', '0.56', '0.56', '0.44', '0.44', '0.50', '0.44', '0.50', '0.39'],
    "S13": ['0.56', '0.50', '0.56', '0.50', '0.56', '0.39', '0.39', '0.39', '0.33', '0.61'],
    "S14": ['0.17', '0.61', '0.33', '0.17', '0.39', '0.33', '0.56', '0.44', '0.50', '0.44'],
    }
    fc2 = {key: [float(value) for value in values] for key, values in fc2.items()}

    fc3 = {
    "S01": ['0.44', '0.44', '0.33', '0.39', '0.39', '0.33', '0.39', '0.33', '0.39', '0.39'],
    "S02": ['0.72', '0.72', '0.67', '0.83', '0.67', '0.56', '0.67', '0.39', '0.67', '0.67'],
    "S03": ['0.56', '0.61', '0.72', '0.67', '0.61', '0.67', '0.56', '0.61', '0.61', '0.61'],
    "S04": ['0.28', '0.06', '0.00', '0.00', '0.06', '0.11', '0.00', '-0.22', '0.06', '-0.11'],
    "S05": ['-0.06', '0.33', '0.28', '0.22', '0.22', '0.17', '0.06', '0.17', '0.28', '0.22'],
    "S06": ['0.50', '0.56', '0.72', '0.61', '0.50', '0.39', '0.33', '0.39', '0.50', '0.50'],
    "S07": ['0.56', '0.56', '0.50', '0.50', '0.50', '0.39', '0.39', '0.44', '0.33', '0.39'],
    "S08": ['0.56', '0.56', '0.61', '0.56', '0.56', '0.50', '0.56', '0.44', '0.67', '0.61'],
    "S09": ['0.61', '0.44', '0.72', '0.56', '0.61', '0.61', '0.56', '0.56', '0.50', '0.61'],
    "S10": ['0.78', '0.72', '0.67', '0.67', '0.67', '0.61', '0.67', '0.78', '0.67', '0.72'],
    "S11": ['0.17', '-0.11', '0.06', '0.06', '0.39', '0.50', '0.17', '0.22', '0.28', '0.28'],
    "S12": ['0.44', '0.39', '0.56', '0.44', '0.39', '0.44', '0.44', '0.39', '0.44', '0.44'],
    "S13": ['0.39', '0.56', '0.50', '0.50', '0.50', '0.44', '0.39', '0.50', '0.39', '0.56'],
    "S14": ['0.17', '0.22', '0.28', '0.50', '0.39', '0.50', '0.50', '0.28', '0.56', '0.50'],
    }
    fc3 = {key: [float(value) for value in values] for key, values in fc3.items()}


    fc1_data = [fc1[subj] for subj in subjects]
    fc2_data = [fc2[subj] for subj in subjects]
    fc3_data = [fc3[subj] for subj in subjects]
    conv_fc_data = [conv_fc[subj] for subj in subjects]
    # conv_bn_fc_data = [conv_bn_fc[subj] for subj in subjects]
    conv34_fc_data = [conv34_fc[subj] for subj in subjects]
    conv234_fc_data = [conv234_fc[subj] for subj in subjects]
    convAll_fc_data = [convAll_fc[subj] for subj in subjects]

    # Create boxplot
    plt.figure(figsize=(20, 8))
    positions = np.arange(len(subjects))

    # Plot each dataset with an offset
    plt.boxplot(fc3_data, positions=positions, widths=width, patch_artist=True, boxprops=dict(facecolor="dimgray"), medianprops=dict(color="black"))
    plt.boxplot(fc2_data, positions=positions + width, widths=width, patch_artist=True, boxprops=dict(facecolor="brown"), medianprops=dict(color="black"))
    plt.boxplot(fc1_data, positions=positions + 2 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="olive"), medianprops=dict(color="black"))
    plt.boxplot(conv_fc_data, positions=positions + 3 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="lightblue"), medianprops=dict(color="black"))
    plt.boxplot(conv34_fc_data, positions=positions + 4 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="purple"), medianprops=dict(color="black"))
    plt.boxplot(conv234_fc_data, positions=positions + 5 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="blue"), medianprops=dict(color="black"))
    plt.boxplot(convAll_fc_data, positions=positions + 6 * width, widths=width, patch_artist=True, boxprops=dict(facecolor="gold"), medianprops=dict(color="black"))
    
    # Create legend manually
    fc3_patch = mpatches.Patch(color='dimgray', label='FC3')
    fc2_patch = mpatches.Patch(color='brown', label='FC2+')
    fc1_patch = mpatches.Patch(color='olive', label='FC1+')
    conv_fc_patch = mpatches.Patch(color='lightblue', label='Conv4+')
    conv34_fc_patch = mpatches.Patch(color='purple', label='Conv3+')
    conv234_fc_patch = mpatches.Patch(color='blue', label='Conv2+')
    convAll_fc_patch = mpatches.Patch(color='gold', label='All-layers')

    
    ### for horizontal line to show mean
    # Compute the mean accuracy across all subjects for both FC-Finetuning and Conv-FC-Finetuning
    fc3_means = [sum(values) / len(values) for values in fc3_data]
    fc2_means = [sum(values) / len(values) for values in fc2_data]
    fc1_means = [sum(values) / len(values) for values in fc1_data]
    conv_fc_means = [sum(values) / len(values) for values in conv_fc_data]
    conv34_fc_means = [sum(values) / len(values) for values in conv34_fc_data]
    conv234_fc_means = [sum(values) / len(values) for values in conv234_fc_data]
    convAll_fc_means = [sum(values) / len(values) for values in convAll_fc_data]

    overall_fc3_mean = sum(fc3_means) / len(fc3_means)
    overall_fc2_mean = sum(fc2_means) / len(fc2_means)
    overall_fc1_mean = sum(fc1_means) / len(fc1_means)
    overall_conv_fc_mean = sum(conv_fc_means) / len(conv_fc_means)
    overall_conv34_fc_mean = sum(conv34_fc_means) / len(conv34_fc_means)
    overall_conv234_fc_mean = sum(conv234_fc_means) / len(conv234_fc_means)
    overall_convAll_fc_mean = sum(convAll_fc_means) / len(convAll_fc_means)

    # Plot horizontal average lines
    plt.axhline(overall_fc3_mean, color='dimgray', linestyle='--', linewidth=1, label='FC3+ Avg')
    plt.axhline(overall_fc2_mean, color='brown', linestyle='--', linewidth=1, label='FC2+ Avg')
    plt.axhline(overall_fc1_mean, color='olive', linestyle='--', linewidth=1, label='FC1+ Avg')
    plt.axhline(overall_conv_fc_mean, color='lightblue', linestyle='--', linewidth=1, label='Conv4+ Avg')
    plt.axhline(overall_conv34_fc_mean, color='purple', linestyle='--', linewidth=1, label='Conv3+ Avg')
    plt.axhline(overall_conv234_fc_mean, color='blue', linestyle='--', linewidth=1, label='Conv3+ Avg')
    plt.axhline(overall_convAll_fc_mean, color='gold', linestyle='--', linewidth=1, label='All-layers Avg')

    # X-axis settings
    plt.xticks(positions + (3 * width), subjects)  # Adjust labels to center
    plt.xlabel("Subjects")
    plt.ylabel("K-Cohen")
    plt.legend(handles=[fc3_patch, fc2_patch, fc1_patch, conv_fc_patch, conv34_fc_patch, conv234_fc_patch, convAll_fc_patch])
    plt.grid(False)
    plt.show()



if __name__ == "__main__":
    main()