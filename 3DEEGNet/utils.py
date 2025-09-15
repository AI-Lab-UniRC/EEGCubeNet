import torch
import math
import mat73
import os
import re
import time
import cv2
import h5py
import timm
import numpy as np
import scipy.io as sio
import random
import netron
import torch.onnx
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.amp import GradScaler, autocast
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset, random_split, Subset, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.model_selection import KFold
import torch
import os
import time
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For GPUs
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False     # Disable CuDNN auto-tuning


def evaluate1(model, classes, test_loader, criterion):
    """
    This function evaluate the model on the test set and returns the average loss, accuracy, kappa amd classification report.

    """   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.permute(0, 2, 1, 3, 4) # only for 3D-CNN
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0, output_dict=True) # ['HC', 'HO', 'RE']
    return avg_loss, accuracy, kappa, report

def train_and_test(model, classes, scaler, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loss_epoch = []
    val_loss_epoch = []
    train_accuracy_epoch = []
    val_accuracy_epoch = []
    # Train for the given number of epochs
    for epoch in range(num_epochs):
        st = time.time()
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for videos, labels in train_loader:
            videos = videos.permute(0, 2, 1, 3, 4)
            # print("shape of input in training -----", videos.shape)
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            # videos = videos.permute(0, 2, 1, 3, 4) # this permute shape is only valid for 3DCNN
            
            
            with torch.amp.autocast('cuda'):

                outputs = model(videos)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Update parameters

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # scheduler.step()

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_kappa = cohen_kappa_score(all_labels, all_preds)

        # Validate after each epoch
        val_loss, val_accuracy, val_kappa, report = evaluate1(model, classes, val_loader, criterion)

        # Append the metrics to lists
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)
        train_accuracy_epoch.append(train_accuracy)
        val_accuracy_epoch.append(val_accuracy)


        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Training Cohen Kappa: {train_kappa:.4f}, Validation Cohen Kappa: {val_kappa:.4f}")
        # print("cls report", report)
        epochtime = time.time() - st
        #log_file.write(f'Fold {fold+1}/{n_splits}, Epoch {epoch+1}/{num_epochs},  '
         #              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Training Cohen Kappa: {train_kappa:.4f}, Validation Cohen Kappa: {val_kappa:.4f} '
         #              f'Epoch Time: {epochtime:.2f}s\n')
    return train_loss_epoch, val_loss_epoch, train_accuracy_epoch, val_accuracy_epoch

def extract_metrics(report):
    accuracy = report['accuracy'] * 100
    precision = report['macro avg']['precision'] * 100
    recall = report['macro avg']['recall'] * 100
    f1 = report['macro avg']['f1-score'] * 100
    return {'accuracy': accuracy, 'precision_macro': precision, 'recall_macro': recall, 'f1_macro': f1}

def compute_avg_std(fold_metrics):
    avg_metrics = {}
    for metric, values in fold_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        avg_metrics[metric] = f"{mean:.2f} ± {std:.2f}"
    return avg_metrics


class ReadVideoDataset(Dataset):
    """
    This class reads the videos from the specific directory and perform preprocessing
    to make sure the image width, height and channels. Then it transform into a tensor of videos and labels
    for further usage. Also, it normalises  the video tensors by 255 to get tensor values in 0 to 1.
    """
    def __init__(self, data_dir, classes, frames=128, height=59, width=59):
        self.data_dir = data_dir
        self.classes = classes
        self.frames = frames
        self.height = height
        self.width = width
        self.video_files = []
        self.labels = []

        # Prepare video file paths and labels
        for label, cls in enumerate(classes):
            class_dir = os.path.join(data_dir, cls)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist!")
                continue
            files = os.listdir(class_dir)
            print(f"Found {len(files)} files in {class_dir}.")  # Debug print
            for file in files:
                if file.endswith('.mp4') or file.endswith('.avi'):
                    video_path = os.path.join(class_dir, file)
                    self.video_files.append(video_path)
                    self.labels.append(label)
                    # print(f"Added file: {video_path}")  # Debug print

        print(f"Total video files loaded: {len(self.video_files)}")  # Debug print
        # print(f"debug to see labels:{self.labels}") # Debug print

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label = self.labels[idx]

        # Load and process the video
        video_tensor = self.load_video(video_file)

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def load_video(self, video_file):
        # cap = cv2.VideoCapture(video_file)

        # frames_list = []
        # frame_count = 0
        # while frame_count < self.frames:
        #     ret, frame = cap.read()
        #     print("in while", ret)
        #     plt.imshow(frame)
        #     if not ret:
        #         break

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Failed to open video:", video_file)
        # else:
        #    print("Video opened successfully")

        frames_list = []
        frame_count = 0
        while frame_count < self.frames:
            ret, frame = cap.read()
            # print(f"Reading frame {frame_count}: Success={ret}")
            if not ret:
                print("End of video or error reading frame.")
                break

            frame = cv2.resize(frame, (self.width, self.height))
            # plt.imshow(frame)
            # Add a channel dimension (grayscale video so it has only 1 channel)
            frame = np.expand_dims(frame, axis=0)

            # If the frame has an extra channel (like 3 for RGB), remove it
            if frame.shape[-1] == 3:
                frame = frame[..., 0]  # Keep only one channel (grayscale)
            # plt.imshow(frame)
            # print("frame shape", frame.shape)
            frames_list.append(frame)
            frame_count += 1

        cap.release()

        # Pad video with empty frames if it has fewer than required frames
        while len(frames_list) < self.frames:
            frames_list.append(np.zeros((1, self.height, self.width))) 

        # Convert list to numpy array and normalize to [0, 1]
        video_tensor = np.array(frames_list, dtype=np.float32) / 255.0

        return torch.tensor(video_tensor)


class ReadVideoDatasetAllSubjects(Dataset):
    """
    This class reads videos from the specified base directory and performs preprocessing 
    to make sure each video has the correct dimensions. It normalizes the video tensors by 255 
    to get tensor values in the range [0, 1].
    It reads all subjects data except the one passed as an argument (subject=subject)
    """
    def __init__(self, base_dir, classes, subject=None, frames=128, height=59, width=59):
        self.base_dir = base_dir
        self.classes = classes
        self.frames = frames
        self.height = height
        self.width = width
        self.video_files = []
        self.labels = []

        # Traverse through each subfolder in the base directory and collect videos
        for folder in os.listdir(base_dir):
            if folder != subject:
                folder_path = os.path.join(base_dir, folder)
                if os.path.isdir(folder_path):  # Check if it's a directory
                    self.load_videos_from_folder(folder_path)

        print(f"Total video files loaded: {len(self.video_files)}")  # Debug print

    def load_videos_from_folder(self, folder_path):
        """
        Helper function to load videos from a specific folder path.
        """
        for label, cls in enumerate(self.classes):
            class_dir = os.path.join(folder_path, cls)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist!")
                continue
            files = os.listdir(class_dir)
            print(f"Found {len(files)} files in {class_dir}.")  # Debug print
            for file in files:
                if file.endswith('.mp4') or file.endswith('.avi'):
                    video_path = os.path.join(class_dir, file)
                    self.video_files.append(video_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label = self.labels[idx]
        video_tensor = self.load_video(video_file)
        return video_tensor, torch.tensor(label, dtype=torch.long)

    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Failed to open video:", video_file)

        frames_list = []
        frame_count = 0
        while frame_count < self.frames:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            # Resize to the specified height and width
            frame = cv2.resize(frame, (self.width, self.height))
            frame = np.expand_dims(frame, axis=0)  # Add a channel dimension (grayscale)

            # If the frame has multiple channels (e.g., RGB), keep only the first channel
            if frame.shape[-1] == 3:
                frame = frame[..., 0]

            frames_list.append(frame)
            frame_count += 1

        cap.release()

        # Pad video with empty frames if fewer than required frames
        while len(frames_list) < self.frames:
            frames_list.append(np.zeros((1, self.height, self.width)))

        # Convert list to numpy array and normalize to [0, 1]
        video_tensor = np.array(frames_list, dtype=np.float32) / 255.0
        return torch.tensor(video_tensor)


def split_data(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    """
    This function split the video dataset into three sets and make sure the class balance in each split.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Extract labels for stratification
    labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])  # Collect labels from dataset
    

    # Split indices for train and temp (val + test)
    stratified_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=random_seed)
    train_idx, temp_idx = next(stratified_split_1.split(range(len(dataset)), labels.numpy()))

    # Generate a temporary label list for stratification on val and test
    temp_labels = labels[temp_idx]

    # Calculate proportions of val and test within temp set
    val_ratio_temp = val_ratio / (val_ratio + test_ratio)
    
    # Split indices within temp for val and test
    stratified_split_2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio_temp, random_state=random_seed)
    val_idx, test_idx = next(stratified_split_2.split(temp_idx, temp_labels.numpy()))

    # Map temp indices back to original dataset indices
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_subset)}")
    
    return train_subset, val_subset, test_subset


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):  
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score

            self.counter = 0

    #def save_checkpoint(self, val_loss, model):
    #    if self.verbose:
    #        print(f"Validation loss decreased. Saving model...")
    #   torch.save(model.state_dict(), self.path)

# early_stopping = EarlyStopping(patience=5, verbose=True, min_delta=0.01)


def evaluate(model, val_loader, criterion):
    """
    Evaluate the model on the validation set.
    Args:
        model: The PyTorch model to be evaluated.
        val_loader: DataLoader for the validation set.
        criterion: Loss function.
    Returns:
        Average validation loss and accuracy.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['HC', 'HO', 'RE'], zero_division=0) # ['CL1', 'CL2'], ['HC', 'HO', 'RE']
    return avg_loss, accuracy, kappa, report

