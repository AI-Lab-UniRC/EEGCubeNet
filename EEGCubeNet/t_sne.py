import torch
import os
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from read_eeg_datasets import ReadVideoDataset
from eegcubenet import Deep3DCNN



def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:  # Assuming a DataLoader is used
            inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
            feature_map = model(inputs, extract_features=True)  # Extract intermediate features
            features.append(feature_map.view(feature_map.shape[0], -1).cpu().numpy())  # Flatten
            labels.append(targets.cpu().numpy())
    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels

def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_features = tsne.fit_transform(features)

    colors = ["blue", "darkred"]  # Class 0 -> Blue, Class 1 -> Red
    cmap = mcolors.ListedColormap(colors)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap=cmap, alpha=0.7)
    plt.colorbar(scatter)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(2)]
    plt.legend(handles, ["HO", "RE"], title="Classes")
    plt.title("All-Layers (Conv1+)", fontsize=24)
    plt.xlabel("t-SNE Dim_1", fontsize=14)
    plt.ylabel("t-SNE Dim_2", fontsize=14)
    plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subject = "S02"
no_classes = 2
batch_size = 8
target_dir = f"/home/neuronas/Motion_Planning_Upper_Limb_Dataset/WEnergy_movies_59x59x128/{subject}"
classes_target = ['HO', 'RE']
model_path = f"/home/sufflin/Documents/MotionPlanning/IJNS-2025/3DCNN-LOSO-AllLayers/AllSubjects-vs-Calibrated-{subject}/HORE"
val_dataloader_target = ReadVideoDataset(target_dir, classes_target, frames=128, height=59, width=59)
val_loader_video = torch.utils.data.DataLoader(val_dataloader_target, batch_size=batch_size, shuffle=False)

model_tsne = Deep3DCNN(no_classes, extract_features=True).to(device)
model_tsne.load_state_dict(torch.load(os.path.join(model_path, 'Finetuned_3DCNN.pth')))

features_tsne, labels_tsne = extract_features(model_tsne, val_loader_video)

plot_tsne(features_tsne, labels_tsne)
