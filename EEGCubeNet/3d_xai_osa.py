

import torch
import os

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# matplotlib.use('Agg')  # Using the Agg backend for cretion of gif
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from statsmodels.stats.anova import AnovaRM
import scipy
import scipy.io as sio
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

from utils import split_data_traintest
from read_eeg_datasets import ReadVideoDataset
from eegcubenet import Deep3DCNN



def occlusion_sensitivity_analysis_sequential_with_probs(model, dataloader, temporal_size=32, spatial_size=10):
    results = []

    for batch_videos, batch_labels in dataloader:
        # Move videos and labels to the same device as the model
        print("Shape of batch_videos and batch_labels:", batch_videos.shape, batch_labels.shape)
        device = next(model.parameters()).device
        batch_videos = batch_videos.to(device)
        batch_labels = batch_labels.to(device)

        batch_size, temporal_dim, _, height, width = batch_videos.shape

        # Ensure temporal_size and spatial_size are valid
        if temporal_size > temporal_dim:
            raise ValueError(f"Temporal size {temporal_size} exceeds temporal dimension {temporal_dim}.")
        if spatial_size > height or spatial_size > width:
            raise ValueError(
                f"Spatial size {spatial_size} exceeds spatial dimensions {height}x{width}."
            )

        # Get the original predictions for comparison
        original_videos = batch_videos.permute(0, 2, 1, 3, 4)  # Permute for model input (B, C, Depth, W, H)
        original_outputs = model(original_videos)
        original_probs = F.softmax(original_outputs, dim=1)  # Probabilities for original videos

        # Loop over temporal slices
        for start_frame in range(0, temporal_dim, temporal_size):
            end_frame = min(start_frame + temporal_size, temporal_dim)

            # Loop over spatial slices (height and width)
            for x_start in range(0, height, spatial_size):
                x_end = min(x_start + spatial_size, height)

                for y_start in range(0, width, spatial_size):
                    y_end = min(y_start + spatial_size, width)

                    # Mask the selected region
                    masked_videos = batch_videos.clone()
                    masked_videos[:, start_frame:end_frame, :, x_start:x_end, y_start:y_end] = 0
                    
                    # Evaluate model on masked videos
                    masked_videos = masked_videos.permute(0, 2, 1, 3, 4)  # Permute for model input (B, C, Depth, W, H)
                    masked_videos = masked_videos[:1, :, :, :, :]
                    masked_outputs = model(masked_videos)
                    masked_labels = torch.argmax(masked_outputs, dim=1)
                    masked_probs = F.softmax(masked_outputs, dim=1)  # Get probability scores for masked videos

                    # Compute relevance by comparing masked probabilities with original probabilities
                    relevance = torch.abs(masked_probs - original_probs)  # Absolute change in probabilities
                    # Record results
                    for i in range(batch_size):
                        results.append({
                            "masked_video_index": i,
                            "true_label": batch_labels[i].item(),
                            "masked_label": masked_labels[i].item(),
                            "original_prob": original_probs[i].tolist(),
                            "masked_prob": masked_probs[i].tolist(),
                            "relevance": relevance[i].tolist(),
                            "masking_region": {
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "x_start": x_start,
                                "x_end": x_end,
                                "y_start": y_start,
                                "y_end": y_end
                            }
                        })
                        break

        break  
    return results


def occlusion_sensitivity_analysis_single_video_with_probs(model, video, temporal_size=64, spatial_size=16, label=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    model.eval()
    model = model.to(device)
    
    # for video in videos:
    video = torch.Tensor(video)

    video = video.unsqueeze(1)
  
    original_video = video
    # temporal_dim, _, height, width = video.shape[1:]
    _, _, temporal_dim, height, width = video.shape
    # Ensure temporal_size and spatial_size are valid
    if temporal_size > temporal_dim:
        raise ValueError(f"Temporal size {temporal_size} exceeds temporal dimension {temporal_dim}.")
    if spatial_size > height or spatial_size > width:
        raise ValueError(
            f"Spatial size {spatial_size} exceeds spatial dimensions {height}x{width}."
        )
    original_video = original_video.to(device)
    original_output = model(original_video)
    original_label = torch.argmax(original_output, dim=1)
    if original_label != label:
        original_label = label
    else:
        original_label = original_label.item()
    # print("inside masking original output and label", original_output, original_label, original_label.item())
    original_prob = F.softmax(original_output, dim=1)  # Probabilities for the original video

    # Loop over temporal slices
    for start_frame in range(0, temporal_dim, temporal_size//2):
        end_frame = min(start_frame + temporal_size, temporal_dim)

        # Loop over spatial slices (height and width)
        for x_start in range(0, height, spatial_size//2):
            x_end = min(x_start + spatial_size, height)

            for y_start in range(0, width, spatial_size//2):
                y_end = min(y_start + spatial_size, width)

                # Mask the selected region
                masked_video = original_video.clone()
                # masked_video[:, start_frame:end_frame, :, x_start:x_end, y_start:y_end] = 0
                masked_video[:, :, start_frame:end_frame, x_start:x_end, y_start:y_end] = 0
                # print("masked video shape", masked_video.shape)

                # Evaluate model on masked video
                # masked_video = masked_video.permute(0, 2, 1, 3, 4)  # Permute for model input (B, C, Depth, W, H)
                masked_video = masked_video.to(device)
                masked_output = model(masked_video)
                masked_label = torch.argmax(masked_output, dim=1)
                masked_prob = F.softmax(masked_output, dim=1)  # Get probability scores for masked video

                # Compute relevance by comparing masked probabilities with original probabilities
                #relevance = torch.abs(masked_prob - original_prob).squeeze(0).tolist()  # Absolute change in probabilities
                relevance = torch.sum(torch.abs(masked_prob - original_prob)).item()
                #print("original, maksked, relevance:", original_prob, masked_prob, relevance)
                # Record results for the single video
                results.append({
                    "masked_video_index": 0,  # Since it's a single video
                    "true_label": original_label, # label.item(),
                    "masked_label": masked_label.item(),
                    "original_prob": original_prob.squeeze(0).tolist(),
                    "masked_prob": masked_prob.squeeze(0).tolist(),
                    "relevance": relevance,
                    "masking_region": {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "x_start": x_start,
                        "x_end": x_end,
                        "y_start": y_start,
                        "y_end": y_end
                    }
                })


    return results


def analyze_masking_results(results, temporal_dim, height, width, temporal_size, spatial_size):
    """
    Analyze and plot influential regions based on occlusion sensitivity results.

    Args:
        results (list): List of results with accuracy and masking_region.
        temporal_dim (int): Total number of frames.
        height (int): Spatial height of the video.
        width (int): Spatial width of the video.
        temporal_size (int): Temporal size of the mask.
        spatial_size (int): Spatial size of the mask.
    """
    # Initialize accuracy matrix for temporal and spatial regions
    temporal_regions = (temporal_dim + temporal_size - 1) // temporal_size
    spatial_regions_h = (height + spatial_size - 1) // spatial_size
    spatial_regions_w = (width + spatial_size - 1) // spatial_size

    # Create a 3D array to store accuracy: [temporal, spatial_h, spatial_w]
    accuracy_matrix = np.zeros((temporal_regions, spatial_regions_h, spatial_regions_w))
    counts = np.zeros_like(accuracy_matrix)  # To calculate average if multiple overlaps

    # Populate accuracy_matrix based on results
    for res in results:
        acc = res["accuracy"]
        region = res["masking_region"]

        t_idx = region["start_frame"] // temporal_size
        h_idx = region["x_start"] // spatial_size
        w_idx = region["y_start"] // spatial_size

        accuracy_matrix[t_idx, h_idx, w_idx] += acc
        counts[t_idx, h_idx, w_idx] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    accuracy_matrix /= counts

    # Aggregate spatial dimensions for easier visualization (temporal x spatial influence)
    spatial_influence = accuracy_matrix.mean(axis=0)
    
    # Plot the temporal-spatial heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(spatial_influence, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Accuracy")
    plt.title("Influence of Masking Regions (Lower Accuracy = Higher Influence)")
    plt.xlabel("Spatial Regions (Width)")
    plt.ylabel("Spatial Regions (Height)")
    plt.xticks(ticks=np.arange(spatial_regions_w), labels=np.arange(spatial_regions_w))
    plt.yticks(ticks=np.arange(spatial_regions_h), labels=np.arange(spatial_regions_h))
    plt.savefig('spat-influence.png')
    plt.show()
    

    # Plot the temporal influence separately
    temporal_influence = accuracy_matrix.mean(axis=(1, 2))
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(temporal_regions), temporal_influence, marker="o", linestyle="-")
    plt.title("Temporal Influence of Masking Regions")
    plt.xlabel("Temporal Regions (Frames)")
    plt.ylabel("Average Accuracy")
    plt.grid()
    plt.savefig('temp-influence.png')
    plt.show()


def rank_relevance(masked_data):
    """
    Rank the relevance of masks where the true label and masked label differ.
    
    Args:
        masked_data (list): List of dictionaries containing masked results.

    Returns:
        ranked_data (list): Ranked data with relevance scores and masking details.
    """
    ranked_data = []

    for idx, entry in enumerate(masked_data):
        # print(entry)
        true_label = entry["true_label"]
        masked_label = entry["masked_label"]
        # print("ranking for video and true and masked labels are:", idx, true_label, masked_label)

        # Only consider masks where labels differ
        if true_label != masked_label:
            original_prob = np.array(entry["original_prob"])
            masked_prob = np.array(entry["masked_prob"])

            # For relevance we can compute KL-divergence, sum of absolute difference, and precited class relevance_score
            relevance = np.sum(np.abs(original_prob - masked_prob))  # Sum of absolute differences
            # KL- divergence
            # Ensure no zero probabilities to avoid log(0)
            # epsilon = 1e-9
            # original_prob = np.clip(original_prob, epsilon, 1)
            # masked_prob = np.clip(masked_prob, epsilon, 1)
            # kl_divergence = np.sum(original_prob * np.log(original_prob / masked_prob))

            # Compute relevance (mean absolute change in probabilities)
            # relevance = np.mean(np.abs(original_prob - masked_prob))

            # Store ranked data
            ranked_data.append({
                "masked_video_index": entry["masked_video_index"],
                "true_label": true_label,
                "masked_label": masked_label,
                "relevance_score": relevance,
                "start_frame": entry["masking_region"]["start_frame"],
                "end_frame": entry["masking_region"]["end_frame"],
                "x_start": entry["masking_region"]["x_start"],
                "x_end": entry["masking_region"]["x_end"],
                "y_start": entry["masking_region"]["y_start"],
                "y_end": entry["masking_region"]["y_end"]
            })

    # Sort ranked data by relevance score (descending)
    ranked_data.sort(key=lambda x: x["relevance_score"], reverse=True)

    return ranked_data


def separate_videos_by_class(data_loader, model, device, save_dir="./separated_videos/"):

    model.eval()  # Set model to evaluation mode
    class_videos = {0: [], 1: []}  # Dictionary to store videos by class
    x = 0
    with torch.no_grad():
        for batch in data_loader:
            videos, labels = batch  # Assuming (videos, labels) in data loader
            videos = videos.to(device)  # Send videos to the same device as the model
            # Predict class probabilities
            videos = videos.permute(0,2,1,3,4)
            outputs = model(videos)
            predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy()
            # print("predcited classes raw batch", predicted_classes)
            x += 1
            print("batch no. ", x)
            print("in the batch following ture labels: ", labels)
            print("in the batch following predicted: ", predicted_classes)
            
            # Process each video in the batch
            for idx, prediction in enumerate(predicted_classes):
                # print("shape of video being saved", videos[idx].shape)
                video = videos[idx].cpu().numpy()  # Convert video to numpy for saving/processing
                
                # Add video to the respective class
                class_videos[prediction].append(video)
                
                # Save video to file if needed
                class_dir = os.path.join(save_dir, f"class_{prediction}")
                os.makedirs(class_dir, exist_ok=True)
                video_filename = os.path.join(class_dir, f"video_{idx}.npy")
                np.save(video_filename, video)
    
    print("Videos separated by predicted classes and saved.")
    return class_videos

def separate_videos(data_loader, device, save_dir="./separated_videos/"):
    class_videos = {0: [], 1: []}
    for batch in data_loader:
            videos, labels = batch  # Assuming (videos, labels) in data loader
            labels = labels.cpu().numpy()
            videos = videos.permute(0,2,1,3,4)
            video = videos[0].cpu().numpy()  # Convert video to numpy for saving/processing
            print("labels are:", labels)

            for l in labels:
                if l == 0:
                    class_videos[l].append(video)
                else:
                    class_videos[l].append(video)
    return class_videos

def find_best_mask(masked_data):
    # Initialize variables to store the best mask details
    # best_accuracy = -1
    best_relevance = -1
    best_mask = None

    # Iterate over masked data to find the best-performing mask
    for idx, entry in enumerate(masked_data):
        #accuracy = entry["accuracy"]
        relevance = entry["relevance"]

        if relevance > best_relevance:
            # best_accuracy = accuracy
            best_relevance = relevance
            best_mask = {
                        "masked_video_index": i,
                        "true_label": batch_labels[i].item(),
                        "original_prob": original_probs[i].tolist(),
                        "masked_prob": probs[i].tolist(),
                        "relevance": relevance[i].tolist(),
                        "masking_region": {
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "x_start": x_start,
                            "x_end": x_end,
                            "y_start": y_start,
                            "y_end": y_end}
                        }
    return best_mask

def plot_temporal_similarity_with_relevance(similarity_counts, class_label):
    temp_similarity = similarity_counts["temporal"]
    print("temporal similarity shape in temp similarity",temp_similarity.keys())
    frame_ranges = [key[:2] for key in temp_similarity.keys()]  # Extract temporal keys
    relevance_scores = list(temp_similarity.values())  # Use relevance-weighted counts
    print("relevance scores", relevance_scores)
    bins = [f"{start}-{end}" for start, end in frame_ranges]
    # print("bins are:", bins[:2])
    # print("relevance scores:", relevance_scores[:2])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bins, relevance_scores, color="blue", alpha=0.7)
    plt.xlabel("Frame Ranges (Temporal Regions)")
    plt.ylabel("Relevance-Weighted")
    plt.title(f"Temporal Similarity (Relevance Weighted) for Class {class_label}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_spatial_similarity_with_relevance(similarity_counts, class_label):
    grid = np.zeros((64, 64))  # Assuming 64x64 spatial resolution
    spa_similarity = similarity_counts["spatial"]

    for key, relevance in spa_similarity.items():
        x_start, x_end, y_start, y_end = key
        grid[x_start:x_end, y_start:y_end] += relevance

    # print("grid shape:", grid)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="jet", interpolation="nearest")
    plt.colorbar(label="Relevance")
    plt.title(f"Spatial Similarity Heatmap (Relevance Weighted) for Class {class_label}")
    plt.xlabel("Frequency")
    plt.ylabel("Channels")
    plt.show()

def channel_wise_relevance_average(similarity_counts, class_label, metric="alpha"):
    ## The metric represent here the average computation mechanism:
    #### alpha: average of first quarter, beta: average of half, and gamma: average of all
      # Assuming 64x64 spatial resolution
    grid = np.zeros((64, 64))
    spa_similarity = similarity_counts["spatial"]

    for key, relevance in spa_similarity.items():
        x_start, x_end, y_start, y_end = key
        grid[x_start:x_end, y_start:y_end] += relevance

    # Discard the last 5 rows and columns to get 59 by 59 shape of channel x xfrequency
    grid = grid[:-5, :-5]

    # # Average all vertical values that are channel frequency values and store in the first column as gamma
    # grid[:, 0] = np.mean(grid, axis=1)

    # Computing different averages for each row based on the lpha, beta, gamma
    if metric=="alpha":
        print("alpha for quarter of channels-frequency")
        q1_avg = np.mean(grid[:, :grid.shape[1] // 4], axis=1)  # First quarter
        grid = q1_avg
        print("grid row is", q1_avg)
        print("grid is", grid)
    if metric=="beta":
        print("beta for quarter of channels-frequency")
        half_avg = np.mean(grid[:, :grid.shape[1] // 2], axis=1)  # First half
        grid = half_avg
    if metric=="gamma":
        print("gamma for quarter of channels-frequency")
        full_avg = np.mean(grid, axis=1)  # Full row
        grid = full_avg
    return grid


def channel_wise_relevance_mean(grid, metric="alpha"):
    ## The metric represent here the average computation mechanism:
    #### alpha: average of first quarter, beta: average of half, and gamma: average of all
    # Assuming 64x64 spatial resolution

    # Discard the last 5 rows and columns to get 59 by 59 shape of channel x xfrequency
    grid = grid[:-5, :-5]

    # # Average all vertical values that are channel frequency values and store in the first column as gamma
    # grid[:, 0] = np.mean(grid, axis=1)

    # Computing different averages for each row based on the lpha, beta, gamma
    if metric=="alpha":
        print("alpha for quarter of channels-frequency")
        q1_avg = np.mean(grid[:, :grid.shape[1] // 4], axis=1)  # First quarter
        grid = q1_avg
        print("grid row is", q1_avg)
        print("grid is", grid)
    if metric=="beta":
        print("beta for quarter of channels-frequency")
        half_avg = np.mean(grid[:, :grid.shape[1] // 2], axis=1)  # First half
        grid = half_avg
    if metric=="gamma":
        print("gamma for quarter of channels-frequency")
        full_avg = np.mean(grid, axis=1)  # Full row
        grid = full_avg

    return grid

def analyze_distribution_and_cluster(band_data, band_name='delta', n_clusters=2, visualize=True):
    """
    band_data: list of 2D arrays (frames × channels) for one band, length = number of tensors (e.g., 18)
    """
    N = len(band_data)
    frames, channels = band_data[0].shape
    flattened = np.array([x.flatten() for x in band_data])  # shape: (N, frames*channels)

    # PCA visualization
    if visualize:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(flattened)
        plt.figure(figsize=(6, 5))
        plt.scatter(reduced[:, 0], reduced[:, 1], c='gray', label='Samples')
        plt.title(f"PCA of {band_name} Band (17 trials)")
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.show()

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(flattened)
    centers = kmeans.cluster_centers_

    # Find largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]

    # Get the index of the sample closest to the cluster center
    indices = np.where(labels == largest_cluster)[0]
    cluster_vectors = flattened[indices]
    center_vector = centers[largest_cluster]
    closest_idx, _ = pairwise_distances_argmin_min(center_vector.reshape(1, -1), cluster_vectors)
    representative_idx = indices[closest_idx[0]]

    representative_matrix = band_data[representative_idx]
    mean_matrix = np.mean(np.stack(band_data), axis=0)

    print(f"[{band_name.upper()}] Representative sample index from cluster: {representative_idx}")

    return {
        'mean': mean_matrix,
        'representative': representative_matrix,
        'cluster_indices': indices,
        'flattened': flattened
    }




batch_size = 8
num_epochs = 15
num_epochs_cnn = 15
n_splits = 10
lr = 0.001
no_classes = 2
subjects =  ['S01'] #['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10','S11', 'S12', 'S13', 'S14'] #'S01', 'S02', 'S03', 'S04', 'S05', 
base_data_dir = '/home/neuronas/Motion_Planning_Upper_Limb_Dataset/WEnergy_movies_59x59x128'
# filename = "/home/sufflin/Documents/MotionPlanning/Results/59x59x128/XAI-4-3DCNN/test_results_zeroFT.txt"

task = 'HORE'
if task == 'HCRE':
    classA = 'HC'
    classB = 'RE'
else:
    classA = 'HO'
    classB = 'RE'

for subject in subjects:
    print(f"Testing Subject: {subject}")
    target_dir = f"{base_data_dir}/{subject}"
    classes_target = ['HO', 'RE']

    save_dir = f"/home/sufflin/Documents/IJNS-GitHub/results/xai/{subject}/{task}"
    model_path = f"/home/sufflin/Documents/IJNS-GitHub/3DCNN-LOSO-Conv3-Conv4-FC/AllSubjects-vs-Calibrated-{subject}/{task}"
    # print(model_path)
    validation_data_target = ReadVideoDataset(target_dir, classes_target, frames=128, height=64, width=64) # when to test on leave-one-out subject data
    train_data_target, val_data_target = split_data_traintest(validation_data_target)
    merged_train_test = train_data_target
    # val_loader_target = torch.utils.data.DataLoader(validation_data_target, batch_size=batch_size, shuffle=False)
    
    trained_model = Deep3DCNN(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = trained_model.to(device)
    trained_model.load_state_dict(torch.load(os.path.join(model_path, 'Finetuned_3DCNN.pth')))
    trained_model.eval()

    # input_tensor = torch.randn(1, 1, 128, 59, 59).to(device)
    # Forward pass
    val_loader_target = torch.utils.data.DataLoader(val_data_target, batch_size=batch_size, shuffle=False)
    for batch_idx, (inputs, labels) in enumerate(val_loader_target):
        print("inputs shape in data_loader", inputs.shape)
        inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
        output = trained_model(inputs)
        print(f"Model output shape: {output.shape}")
        break

    """Splitting the Videos for Class 0 and Class 1"""
    save_dir_sepvid = f"/home/sufflin/Documents/IJNS-GitHub/results/xai/{subject}/{task}/separated_videos/"
    save_dir_maskvid = f"/home/sufflin/Documents/IJNS-GitHub/results/xai/{subject}/{task}/masked_videos/"
    
     
    ## separated_videos = separate_videos_by_class(val_loader_target, trained_model, device="cuda", save_dir=save_dir_sepvid)
    separated_videos = separate_videos(val_loader_target, device="cuda", save_dir=save_dir_sepvid)
    videos_0 = separated_videos[0]
    videos_1 = separated_videos[1]
    print(" length of videos ", len(videos_0), len(videos_1), type(videos_1))

    masked_ranked_videos_0 = []
    masked_ranked_videos_1 = []


    ## optimisation parameters for different masking windows 16, 8
    temporal_sizes = [32, 64] 
    spatial_sizes = [2, 4, 8, 16, 24]

    # Initialize storage for maximum counts and corresponding masks
    max_count_class0 = 0
    max_count_class1 = 0
    masked_ranked_videos_0 = []
    masked_ranked_videos_1 = []

    
    ##  Run the following for loops for videos_0 and videos_1 once, and use them for the rest of procesisng (these are time consuming operations)
    """
    # For Class 0
    for index, video in enumerate(videos_0):
        for temporal_size in temporal_sizes:
            for spatial_size in spatial_sizes:
                masked_data_class0 = occlusion_sensitivity_analysis_single_video_with_probs(
                    trained_model, video, temporal_size=temporal_size, spatial_size=spatial_size, label=0)
                #print(f"Class 0 | Temporal size: {temporal_size}, Spatial size: {spatial_size}, Total masked volumes: {len(masked_data_class0)}")
                ranked_masks_0 = rank_relevance(masked_data_class0)
                #print(ranked_masks_0[:10])
                current_count = len(ranked_masks_0)
                #print(f"Class 0 | Temporal size: {temporal_size}, Spatial size: {spatial_size}, Count of impactful ranked masks: {current_count}")
            
                # 1- Keep track of the maximum count (previous approach)
                if current_count > max_count_class0:
                    max_count_class0 = current_count
                    masked_ranked_videos_0 = []
                    masked_ranked_videos_0.append(ranked_masks_0) 
                elif current_count == max_count_class0:
                    masked_ranked_videos_0 = []
                    masked_ranked_videos_0.append(ranked_masks_0)
                
                # Instead of (1), need to keep all unique masks to append in the masked videos for 0 and 1
                # also, all the ranked masks should be observed whether they already exists in the list or not
                # this by checking simply the frame, x, y start and end values.
                #if len(masked_ranked_videos_0) != 0:
                #    masked_ranked_videos_0.append(ranked_masks_0) 
                #elif ranked_masks_0

        class_dir = os.path.join(save_dir_maskvid, f"masked_videos_0")
        os.makedirs(class_dir, exist_ok=True)
        video_filename = os.path.join(class_dir, f"video_{index}.npy")
        np.save(video_filename, masked_ranked_videos_0) 

    # For Class 1
    for index, video in enumerate(videos_1):
        for temporal_size in temporal_sizes:
            for spatial_size in spatial_sizes:
                masked_data_class1 = occlusion_sensitivity_analysis_single_video_with_probs(
                    trained_model, video, temporal_size=temporal_size, spatial_size=spatial_size, label=1)
                # print(f"Class 1 | Temporal size: {temporal_size}, Spatial size: {spatial_size}, Total masked volumes: {len(masked_data_class1)}")
                ranked_masks_1 = rank_relevance(masked_data_class1)
                current_count = len(ranked_masks_1)
                # print(f"Class 1 | Temporal size: {temporal_size}, Spatial size: {spatial_size}, Count of impactful ranked masks: {current_count}")
                
                # 1- Keep track of the maximum count
                # Instead of (1), need to keep all unique masks to append in the masked videos for 0 and 1
                # also, all the ranked masks should be observed whether they already exists in the list or not
                # this by checking simply the frame, x, y start and end values.

                if current_count > max_count_class1:
                    max_count_class1 = current_count
                    masked_ranked_videos_1 = []
                    masked_ranked_videos_1.append(ranked_masks_1) 
                elif current_count == max_count_class1:
                    masked_ranked_videos_1 = []
                    masked_ranked_videos_1.append(ranked_masks_1)

        class_dir = os.path.join(save_dir_maskvid, f"masked_videos_1")
        os.makedirs(class_dir, exist_ok=True)
        video_filename = os.path.join(class_dir, f"video_{index}.npy")
        np.save(video_filename, masked_ranked_videos_1)

    # Final Output
    print(f"Maximum count for Class 0: {max_count_class0}")
    print(f"Maximum count for Class 1: {max_count_class1}")
    """


    # #  Loading the masked videos for class 0 and 1 to compute relevance.

    masked_ranked_videos_0  = []
    masked_ranked_videos_1  = []

    class_dir = os.path.join(save_dir_maskvid, f"masked_videos_0")
    files_0 = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
    print("file list 0 is here:", files_0)
    for file in files_0:
        video_filename = os.path.join(class_dir, f"{file}")
        loaded_video = np.load(video_filename, allow_pickle=True) 
        masked_ranked_videos_0.append(loaded_video)

    class_dir = os.path.join(save_dir_maskvid, f"masked_videos_1")
    files_0 = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
    print("file list 1 is here:", files_0)
    for file in files_0:
        video_filename = os.path.join(class_dir, f"{file}")
        loaded_video = np.load(video_filename, allow_pickle=True) 
        masked_ranked_videos_1.append(loaded_video)
    

    save_dir_relevance_0 = f"/home/sufflin/Documents/IJNS-GitHub/results/xai/{subject}/{task}/relevance_videos/{classA}/"
    save_dir_relevance_1 = f"/home/sufflin/Documents/IJNS-GitHub/results/xai/{subject}/{task}/relevance_videos/{classB}/"
    
    ## here is the averaging of topoplots and lineplts for subjects
    all_deltas = []
    all_thetas = []
    all_alphas = []
    all_betas = []

    for v, video in enumerate(masked_ranked_videos_0):
            grid_size = (64, 64)
            num_frames = 128
            relevance_tensor = np.zeros((num_frames, *grid_size))  # Shape: (128, 64, 64)
            for mask in video[0]:
                x_start, x_end = mask["x_start"], mask["x_end"]  # Convert to zero-based index
                y_start, y_end = mask["y_start"], mask["y_end"]  # Convert to zero-based index
                frame_start, frame_end = mask["start_frame"], mask["end_frame"]  # Zero-based
                relevance_score = mask["relevance_score"]

                # Iterate over each pixel in the mask's spatial region
                for t in range(frame_start, frame_end):  # Iterate through frames
                    for i in range(y_start, y_end):
                        for j in range(x_start, x_end):
                            relevance_tensor[t, i, j] += relevance_score  # Store spatial relevance for each frame
                            #print("relevance at t, i j", t, i, j)
                            #print("the relevance value", relevance_tensor[t, i, j])

            relevance_tensor = relevance_tensor[:, :-5, :-5]
            relevance_tensor_topo = np.mean(relevance_tensor[:, :, :], axis=2) 
            print("shape of relevance tensor for topoplot", relevance_tensor_topo.shape)
            delta = np.mean(relevance_tensor[:, :, 0:4], axis=2)  
            #delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))
            theta = np.mean(relevance_tensor[:, :, 4:8], axis=2)  
            #theta = (theta - np.min(theta)) / (np.max(theta) - np.min(theta))
            alpha = np.mean(relevance_tensor[:, :, 8:13], axis=2)  
            #alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
            beta = np.mean(relevance_tensor[:, :, 13:40], axis=2)  # Shape: (128, 59) (Averaging over all 59 channels)
            #beta = (beta - np.min(beta)) / (np.max(beta) - np.min(beta))

            all_deltas.append(delta)
            all_thetas.append(theta)
            all_alphas.append(alpha)
            all_betas.append(beta)
    results = {}
    delta_topo = analyze_distribution_and_cluster(all_deltas, 'delta', visualize=False)
    theta_topo = analyze_distribution_and_cluster(all_thetas, 'theta', visualize=False)
    alpha_topo = analyze_distribution_and_cluster(all_alphas, 'alpha', visualize=False)
    beta_topo = analyze_distribution_and_cluster(all_betas, 'beta', visualize=False)

    all_topo = [delta_topo, theta_topo, alpha_topo, beta_topo] #delta_topo, 
    bands = ["Delta band", "Theta band", "Alpha band", "Beta band"] # 


    # # Final global average across band-wise means
    # final_mean_all_bands = np.mean(
    #     np.stack([results[b]['mean'] for b in ['delta', 'theta', 'alpha', 'beta']]), axis=0)
    # print(final_mean_all_bands.shape)

    # Load EEG montage (standard 10-20 system)
    import mne
    montage = mne.channels.make_standard_montage("standard_1005")

    for t, topo in enumerate(all_topo):
        # Extract channel names and positions
        topo = topo['representative']
        ch_names = montage.ch_names  # List of channel names
        ch_names_mp = ["F3", "F1", "Fz", "F2", "F4", "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h", 
                       "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FTT7h", "FCC5h", "FCC3h", "FCC1h", 
                       "FCC2h", "FCC4h", "FCC6h", "FTT8h", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "TTP7h", 
                       "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h", "TTP8h", "CP5", "CP3", "CP1", 
                       "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h", "P3", 
                       "P1", "Pz", "P2", "P4"]

        ch_names_2use = [ch for ch in ch_names if ch in ch_names_mp]
        print("Final channel names", len(ch_names_2use), ch_names_2use)

        # Get channel positions
        ch_pos = montage.get_positions()["ch_pos"]  # Dictionary of {channel_name: (x, y, z)}
        pos = np.array([ch_pos[ch][:2] for ch in ch_pos if ch in ch_names_2use])  # Extract (x, y)
        # Simulated EEG relevance data (Frames x Channels)
        num_frames = 128   # Total time points
        num_channels = len(ch_names_2use)  # Should be 59
        # 
        num_segments = 8   # Define the number of segments (8 segments of 16 frames each)
        frames_per_segment = num_frames // num_segments  # 16 frames per segment
        # Create figure for 8 subplots
        fig, axes = plt.subplots(1, num_segments, figsize=(16, 5))
        for i in range(num_segments):
            start_frame = i * frames_per_segment
            end_frame = start_frame + frames_per_segment
            avg_topo = np.mean(topo[start_frame:end_frame, :], axis=0)  # Average over 16 frames
            # Plot topomap
            mne.viz.plot_topomap(avg_topo[:num_channels], pos, cmap="coolwarm", axes=axes[i], show=False)
            axes[i].set_title(f"Frames {start_frame+1}-{end_frame}")
        plotname = f'a_{subject}_{task}_{classA}_{bands[t]}_topomap_intervals.png'
        fig.savefig(os.path.join(save_dir_relevance_0, plotname), bbox_inches='tight', dpi = 300)

        # fig, axes = plt.figure(figsize=(10, 5))
        
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        avg_topo = np.mean(topo[:, :], axis=0)
        # axes.set_title(f"S01 topomap for RE class")
        mne.viz.plot_topomap(avg_topo[:num_channels], pos, names=ch_names_2use, cmap="coolwarm", axes=axes)
        fig.tight_layout()
        # plt.title(f"EEG {bands[t]} topomap for RE class")
        # fig.suptitle(f"EEG {bands[t]} Topomap for RE class", fontsize=16)
        # plt.show()
        plotname = f'a_{subject}_{task}_{classA}_{bands[t]}_topomap.png'
        fig.savefig(os.path.join(save_dir_relevance_0, plotname), bbox_inches='tight', dpi = 300)

        ##### averaged topomap for a single subject for specific class

        # Compute total relevance per channel (sum over frames and frequencies)
        channel_relevance = np.sum(topo, axis=0)  # Shape: (59,)
        # Identify top-10 channels
        top_10_indices = np.argsort(channel_relevance)[-5:]  # Get indices of 10 highest relevance channels
        print("top 10 indices to beout of index", top_10_indices)
        # Extract relevance scores of these channels over time
        top_10_relevance_over_time = topo[:, top_10_indices]#.sum(axis=1)  # Shape: (128, 10)
        print("top 10 relevance over time array", top_10_relevance_over_time.shape)
        # normalise the relevance 0 to 1 for ploting
        top_10_relevance_over_time = (top_10_relevance_over_time - top_10_relevance_over_time.min(axis=0)) / (top_10_relevance_over_time.max(axis=0) - top_10_relevance_over_time.min(axis=0))
        # Plot the relevance of top-10 channels over time
        fig = plt.figure(figsize=(10, 5))
        for i, channel_idx in enumerate(top_10_indices):
            channel_name = ch_names_2use[channel_idx]  # Get channel name
            #print(" iter and channel id, len of channels", i, channel_idx, len(channel_names), channel_name)
            plt.plot(range(128), top_10_relevance_over_time[:, i], label=f'{channel_name}')
        plt.xlabel('Frames (Time)')
        plt.ylabel('Relevance Score')
        #plt.title('Top-5 Channels Relevance Over Time: Class HC')
        plt.legend()
        plt.grid(False)
        # plt.show()
        plotname = f'a_{subject}_{task}_{classA}_{bands[t]}_lineplot.png'
        fig.savefig(os.path.join(save_dir_relevance_0, plotname), bbox_inches='tight', dpi = 300)
    
    # for a in range(5):
    #     print(x)
    ##### Multiple analysis to compute the results for trial-level, spatial-relevance, temporal-relevance, and frequency bands-level for those plots. 
    
    # ##### HC or HO for masked_0
    # for v, video in enumerate(masked_ranked_videos_0):
    #     grid_size = (64, 64)
    #     num_frames = 128
    #     relevance_tensor = np.zeros((num_frames, *grid_size))  # Shape: (128, 64, 64)
    #     for mask in video[0]:
    #         x_start, x_end = mask["x_start"], mask["x_end"]  # Convert to zero-based index
    #         y_start, y_end = mask["y_start"], mask["y_end"]  # Convert to zero-based index
    #         frame_start, frame_end = mask["start_frame"], mask["end_frame"]  # Zero-based
    #         relevance_score = mask["relevance_score"]

    #         # Iterate over each pixel in the mask's spatial region
    #         for t in range(frame_start, frame_end):  # Iterate through frames
    #             for i in range(y_start, y_end):
    #                 for j in range(x_start, x_end):
    #                     relevance_tensor[t, i, j] += relevance_score  # Store spatial relevance for each frame
    #                     #print("relevance at t, i j", t, i, j)
    #                     #print("the relevance value", relevance_tensor[t, i, j])

    #     relevance_tensor = relevance_tensor[:, :-5, :-5]
    #     relevance_tensor_topo = np.mean(relevance_tensor[:, :, :], axis=2) 
    #     print("shape of relevance tensor for topoplot", relevance_tensor_topo.shape)
    #     delta = np.mean(relevance_tensor[:, :, 0:4], axis=2)  
    #     delta = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))
    #     theta = np.mean(relevance_tensor[:, :, 4:8], axis=2)  
    #     theta = (theta - np.min(theta)) / (np.max(theta) - np.min(theta))
    #     alpha = np.mean(relevance_tensor[:, :, 8:13], axis=2)  
    #     alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
    #     beta = np.mean(relevance_tensor[:, :, 13:40], axis=2)  # Shape: (128, 59) (Averaging over all 59 channels)
    #     beta = (beta - np.min(beta)) / (np.max(beta) - np.min(beta))

        # save_rel_vect_mat = os.path.join(save_dir_relevance_0, f'relevance_time_channel_band_{v}.mat') 
        # # Save data to .mat file.... todo: yet to normalise the values for the the mat vector.
        # scipy.io.savemat(save_rel_vect_mat, {"delta": delta, "theta": theta, "alpha": alpha, "beta": beta})

        # #### Task-0
        # ## band wise channel spatial relevance
        # # Define frequency bands (assume your frequency axis is indexed appropriately)
        # freq_bands = {
        #     "Delta (0-4 Hz)": (0, 4),   # Example: First 4 frequency bins
        #     "Theta (4-8 Hz)": (4, 8),
        #     "Alpha (8-13 Hz)": (8, 13),
        #     "Beta (13-40 Hz)": (13, 40)
        # }
        # num_channels = 59
        # channel_names = ["F3", "F1", "Fz", "F2", "F4", "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h", 
        #                  "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FTT7h", "FCC5h", "FCC3h", "FCC1h", 
        #                  "FCC2h", "FCC4h", "FCC6h", "FTT8h", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "TTP7h", 
        #                  "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h", "TTP8h", "CP5", "CP3", "CP1", 
        #                  "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h", "P3", 
        #                  "P1", "Pz", "P2", "P4"]
        # num_bands = len(freq_bands)
        # band_relevance = np.zeros((num_channels, num_bands))  # 59 channels × 4 bands

        # # Loop over channels and compute relevance for each frequency band
        # for ch in range(num_channels):
        #     for i, (band_name, (f_start, f_end)) in enumerate(freq_bands.items()):
        #         band_relevance[ch, i] = np.mean(relevance_tensor[:, ch, f_start:f_end])  # Average over time & freq

        # band_relevance = (band_relevance - np.min(band_relevance)) / (np.max(band_relevance) - np.min(band_relevance))
        # # Plot heatmap
        # fig = plt.figure(figsize=(8, 10))
        # sns.heatmap(
        #     band_relevance,
        #     cmap="jet",
        #     xticklabels=freq_bands.keys(),  # Frequency band names on x-axis
        #     yticklabels=channel_names,  # EEG channel names on y-axis
        #     annot=False, fmt=".0f", linewidths=0.2, cbar=True
        #     )
        # plt.xlabel("Frequency Bands")
        # plt.ylabel("EEG Channels")
        # #plt.title("Spatial Relevance of Channels & Frequency Bands (Averaged Over Time): Class HC")
        # #plt.show()
        # plotname = f'{classA}_trial_{v}_spatial_bands.png'
        # # fig.savefig(os.path.join(save_dir_relevance_0, plotname), bbox_inches='tight', dpi = 300)

         
        ### Task-1

        # # Compute total relevance per channel (sum over frames and frequencies)
        # channel_relevance = np.sum(relevance_tensor, axis=(0, 2))  # Shape: (59,)
        # # Identify top-10 channels
        # top_10_indices = np.argsort(channel_relevance)[-5:]  # Get indices of 10 highest relevance channels
        # print("top 10 indices to beout of index", top_10_indices)
        # # Extract relevance scores of these channels over time
        # top_10_relevance_over_time = relevance_tensor[:, top_10_indices, :].sum(axis=2)  # Shape: (128, 10)
        # # normalise the relevance 0 to 1 for ploting
        # top_10_relevance_over_time = (top_10_relevance_over_time - top_10_relevance_over_time.min(axis=0)) / (top_10_relevance_over_time.max(axis=0) - top_10_relevance_over_time.min(axis=0))
        # # Plot the relevance of top-10 channels over time
        # fig = plt.figure(figsize=(10, 5))
        # for i, channel_idx in enumerate(top_10_indices):
        #     channel_name = channel_names[channel_idx]  # Get channel name
        #     #print(" iter and channel id, len of channels", i, channel_idx, len(channel_names), channel_name)
        #     plt.plot(range(128), top_10_relevance_over_time[:, i], label=f'{channel_name}')
        # plt.xlabel('Frames (Time)')
        # plt.ylabel('Relevance Score')
        # #plt.title('Top-5 Channels Relevance Over Time: Class HC')
        # plt.legend()
        # plt.grid(False)
        # #plt.show()
        # plotname = f'{classA}_trial_{v}_top_channels_lineplot.png'
        # # fig.savefig(os.path.join(save_dir_relevance_0, plotname), bbox_inches='tight', dpi = 300)
        
        
        # ## Task-2
        # # Compute total relevance per channel (sum over frames and frequencies)
        # channel_relevance = np.sum(relevance_tensor, axis=(0, 2))  # Shape: (59,)
        # # Identify top-5 channels
        # top_5_channels = np.argsort(channel_relevance)[-5:]  # Get indices of 5 highest relevance channels
        # top_5_channel_names = [channel_names[i] for i in top_5_channels]
        # #print("top channels and names", top_5_channels, top_5_channel_names)
        # # Extract relevance scores of these channels over time
        # top_5_relevance_over_time = relevance_tensor[:, top_5_channels, :].sum(axis=2)  # Shape: (128, 5)
        # top_5_relevance_over_time = (top_5_relevance_over_time - np.min(top_5_relevance_over_time)) / (np.max(top_5_relevance_over_time) - np.min(top_5_relevance_over_time))
        # #  Plot heatmap
        # fig = plt.figure(figsize=(10, 6))
        # sns.heatmap(top_5_relevance_over_time.T, cmap="viridis", annot=False, cbar=True)
        # plt.xlabel("Frames (Time)")
        # plt.ylabel("Channels")
        # #plt.title(f"Time-Specific Importance of Top-10 Channels: Class HC")
        # plt.yticks(ticks=np.arange(5) + 0.5, labels=[ch for ch in top_5_channel_names])  # Set y-ticks as channel indices
        # #plt.show()
        # plotname = f'{classA}_trial_{v}_top_channels_heatmap.png'
        # # fig.savefig(os.path.join(save_dir_relevance_0, plotname), bbox_inches='tight', dpi = 300)

    #     ### Task-3 fore the topographical plots
    #     
    #     import matplotlib.animation as animation
    #     import mne

    #     # Load EEG montage (standard 10-20 system)
    #     montage = mne.channels.make_standard_montage("standard_1005")

    #     # Extract channel names and positions
    #     ch_names = montage.ch_names  # List of channel names
    #     ch_names_mp = ["F3", "F1", "Fz", "F2", "F4", "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h", 
    #                              "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FTT7h", "FCC5h", "FCC3h", "FCC1h", 
    #                              "FCC2h", "FCC4h", "FCC6h", "FTT8h", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "TTP7h", 
    #                              "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h", "TTP8h", "CP5", "CP3", "CP1", 
    #                              "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h", "P3", 
    #                              "P1", "Pz", "P2", "P4"]
    #     # print("channel names", ch_names)
    #     ch_names_2use = [ch for ch in ch_names if ch in ch_names_mp]
    #     print("final channel names", len(ch_names_2use), ch_names_2use)
    #     ch_pos = montage.get_positions()["ch_pos"]  # Dictionary of {channel_name: (x, y, z)}

    #     # Convert to 2D NumPy array (only x and y coordinates)
    #     pos = np.array([ch_pos[ch][:2] for ch in ch_pos if ch in ch_names_2use]) #ch_names_2use if ch in ch_pos])  # Extract (x, y)
    #     # Simulated EEG relevance data (Frames x Channels)
    #     num_frames = 128   # Total time points
    #     num_channels = len(ch_names_2use)  # Should be 59
    #     # relevance_data = np.random.rand(num_frames, num_channels)  # Replace with real data
    #     freq_bands = ["Delta", "Theta", "Alpha", "Beta"]
    #     # Create figure for animation
    #     fig, ax = plt.subplots(figsize=(6, 5))
    #     def update(frame):
    #         """Update function for animation"""
    #         ax.clear()
    #         mne.viz.plot_topomap(relevance_tensor_topo[frame][:num_channels], pos, cmap="coolwarm", axes=ax, show=True)
    #         ax.set_title(f"EEG Relevance at Frame {frame + 1}")
    #     # Animate topomap over time
    #     ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200)
    #     plt.show()
    #     # topo_hc = os.path.join(save_dir_relevance_1, "RE_topomap_trial0.gif")
    #     # ani.save(filename=topo_hc, writer="pillow")
    #     print(x)

    #     # Load EEG montage (standard 10-20 system)
    #     montage = mne.channels.make_standard_montage("standard_1005")

    #     # Extract channel names and positions
    #     ch_names = montage.ch_names  # List of channel names
    #     ch_names_mp = ["F3", "F1", "Fz", "F2", "F4", "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h", 
    #                    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FTT7h", "FCC5h", "FCC3h", "FCC1h", 
    #                    "FCC2h", "FCC4h", "FCC6h", "FTT8h", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "TTP7h", 
    #                    "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h", "TTP8h", "CP5", "CP3", "CP1", 
    #                    "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h", "P3", 
    #                    "P1", "Pz", "P2", "P4"]

    #     ch_names_2use = [ch for ch in ch_names if ch in ch_names_mp]
    #     print("Final channel names", len(ch_names_2use), ch_names_2use)

    #     # Get channel positions
    #     ch_pos = montage.get_positions()["ch_pos"]  # Dictionary of {channel_name: (x, y, z)}
    #     pos = np.array([ch_pos[ch][:2] for ch in ch_pos if ch in ch_names_2use])  # Extract (x, y)

    #     # Simulated EEG relevance data (Frames x Channels)
    #     num_frames = 128   # Total time points
    #     num_channels = len(ch_names_2use)  # Should be 59
    #     # data = np.random.rand(num_frames, num_channels)  # Replace with real data

    #     # Define the number of segments (8 segments of 16 frames each)
    #     num_segments = 8
    #     frames_per_segment = num_frames // num_segments  # 16 frames per segment

    #     # Create figure for 8 subplots
    #     fig, axes = plt.subplots(1, num_segments, figsize=(20, 5))

    #     for i in range(num_segments):
    #         start_frame = i * frames_per_segment
    #         end_frame = start_frame + frames_per_segment
    #         avg_topo = np.mean(delta[start_frame:end_frame, :], axis=0)  # Average over 16 frames
            
    #         # Plot topomap
    #         mne.viz.plot_topomap(avg_topo[:num_channels], pos, cmap="coolwarm", axes=axes[i], show=False)
    #         axes[i].set_title(f"Frames {start_frame+1}-{end_frame}")

    #     plt.tight_layout()
    #     # plt.title("Channels for a complete trial")
    #     fig.suptitle("EEG Topomap Averaged Over 16-Frame Intervals for a complete trial of RE", fontsize=16)
    #     plt.show()

