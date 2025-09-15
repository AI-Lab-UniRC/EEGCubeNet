
import os
import torch
import time
import random
import numpy as np
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Subset, Subset
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, roc_curve


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For GPUs
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False     # Disable CuDNN auto-tuning


def split_data_matrices(dataset, mapping, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    This function splits the dataset into training, validation, and test sets
    while ensuring class balance using stratified sampling.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Define the label mapping (string labels to integers)
    # label_mapping = {'HC': 0, 'HO': 1, 'RE': 2}  # Update based on your labels
    label_mapping = mapping
    # Extract labels for stratification (convert string labels to integers)
    labels = torch.tensor([label_mapping[dataset[i][1]] if isinstance(dataset[i][1], str) else dataset[i][1] 
                          for i in range(len(dataset))])  # Convert string labels to integer labels

    # Split indices for train and temp (val + test)
    stratified_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=random_seed)
    train_idx, temp_idx = next(stratified_split_1.split(range(len(dataset)), labels.numpy()))

    # Generate a temporary label list for stratification on val and test
    temp_labels = labels[temp_idx]

    # Calculate proportions of val and test within the temp set
    val_ratio_temp = val_ratio / (val_ratio + test_ratio)
    
    # Split indices within temp for val and test
    stratified_split_2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio_temp, random_state=random_seed)
    val_idx, test_idx = next(stratified_split_2.split(temp_idx, temp_labels.numpy()))

    # Map temp indices back to original dataset indices
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]

    # Create subsets using the corresponding indices
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    
    # Print the sizes of the subsets for debugging
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_subset)}")
    
    return train_subset, val_subset, test_subset


def split_data(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    """
    This function split the video dataset into three and make sure the class balance in each split.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Extract labels for stratification
    labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])  # Collect labels from dataset, only commented for matrices and when labels in list
    
    # print(labels[0], len(labels)) # debug

    # Split indices for train and temp (val + test)
    stratified_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=random_seed)
    train_idx, temp_idx = next(stratified_split_1.split(range(len(dataset)), labels.numpy()))
    #print("train idx",train_idx)
    # Generate a temporary label list for stratification on val and test
    temp_labels = labels[temp_idx]

    # Calculate proportions of val and test within temp set
    val_ratio_temp = val_ratio / (val_ratio + test_ratio)
    
    # Split indices within temp for val and test
    stratified_split_2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio_temp, random_state=random_seed)
    val_idx, test_idx = next(stratified_split_2.split(temp_idx, temp_labels.numpy()))
    #print("val idx",val_idx)
    #print("test idx",test_idx)
    # Map temp indices back to original dataset indices
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    
    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_subset)}")
    
    return train_subset, val_subset, test_subset


def split_data_traintest(dataset, train_ratio=0.7, test_ratio=0.3, random_seed=42):
    """
    This function splits the video dataset into train and test sets with balanced classes in each split.
    """
    assert train_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Extract labels for stratification
    labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])  # Collect labels from dataset

    # Split indices for train and test
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    train_idx, test_idx = next(stratified_split.split(range(len(dataset)), labels.numpy()))

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    
    print(f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")
    
    return train_subset, test_subset


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):  #path='checkpoint.pt'
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        #path='checkpoint.pt'self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #if self.verbose:
            #   print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    #def save_checkpoint(self, val_loss, model):
    #    if self.verbose:
    #        print(f"Validation loss decreased. Saving model...")
    #   torch.save(model.state_dict(), self.path)

def reinitialize_weights(m):
    """
    The following function is utilised in k-folf cross validation to intialise the weights of the model for each fold validation.
    This covers Conv3D, LSTM amd Transformer model layers to reinitialise the weights.
    """
    set_seed(42)
    # Initialize Conv2d and Linear layers
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    
    # Initalization of conv3D layers weights
    elif isinstance(m, nn.Conv3d) :
        m.weight.data.normal_(0.0,0.001)

    elif isinstance(m, nn.Conv3d):
        m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
    
    elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    # Initialize MultiheadAttention layer
    elif isinstance(m, torch.nn.MultiheadAttention):
        torch.nn.init.xavier_uniform_(m.in_proj_weight)
        torch.nn.init.xavier_uniform_(m.out_proj.weight)
        if m.in_proj_bias is not None:
            torch.nn.init.zeros_(m.in_proj_bias)
        if m.out_proj.bias is not None:
            torch.nn.init.zeros_(m.out_proj.bias)
    
    # Initialize LayerNorm layer
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

    # Initialize Transformer Encoder Layers (which typically have nn.Linear inside)
    elif isinstance(m, torch.nn.TransformerEncoderLayer):
        torch.nn.init.xavier_uniform_(m.linear1.weight)
        torch.nn.init.xavier_uniform_(m.linear2.weight)
        if m.linear1.bias is not None:
            torch.nn.init.zeros_(m.linear1.bias)
        if m.linear2.bias is not None:
            torch.nn.init.zeros_(m.linear2.bias)

    # Initialize LSTM layers
    elif isinstance(m, torch.nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)


def model_testing(model, test_loader, criterion):
    """
    This function evaluate the model on the test set and returns the average loss, accuracy, kappa amd classification report.        
    """   
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in test_loader:
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
    report = classification_report(all_labels, all_preds, target_names=['HC', 'HO', 'RE'], zero_division=0) # ['CL1', 'CL2']
    return avg_loss, accuracy, kappa, report

def plot_filters_and_feature_maps(model, activations, input_image):
    
    # Check the input dimensions
    if input_image.ndim == 3:  # [channels, height, width]
        input_image = input_image.unsqueeze(0)  # Add batch dimension
    elif input_image.ndim == 4:  # Already [batch, channels, height, width]
        pass
    else:
        raise ValueError("Input image should be 3D or 4D tensor.")

    # Run the model on an input image to get activations
    with torch.no_grad():
        _ = model(input_image)  # Add batch dimension

    # Iterate through each layer's activations
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            print(f"\nVisualizing {name}:")
            # Plot filters (weights)
            filters = layer.weight.detach().cpu()
            fig, axs = plt.subplots(1, filters.size(0), figsize=(filters.size(0)*2, 2))
            fig.suptitle(f"{name} Filters", fontsize=16)
            for i, ax in enumerate(axs):
                ax.imshow(filters[i, 0], cmap='gray')
                ax.axis('off')
            plt.show()

            # Plot feature maps (activations)
            feature_map = activations[layer]
            fig, axs = plt.subplots(1, feature_map.size(1), figsize=(feature_map.size(1)*2, 2))
            fig.suptitle(f"{name} Feature Maps", fontsize=16)
            for i, ax in enumerate(axs):
                ax.imshow(feature_map[0, i], cmap='gray')
                ax.axis('off')
            plt.show()


def train_and_test(model, classes, scaler, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loss_epoch = []
    val_loss_epoch = []
    train_accuracy_epoch = []
    val_accuracy_epoch = []
    early_stopping = EarlyStopping(patience=3, verbose=True, delta=0.01)
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
        val_loss, val_accuracy, val_kappa, report, roc = evaluate(model, classes, val_loader, criterion)

        # Append the metrics to lists
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)
        train_accuracy_epoch.append(train_accuracy)
        val_accuracy_epoch.append(val_accuracy)


        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Training Cohen Kappa: {train_kappa:.4f}, Validation Cohen Kappa: {val_kappa:.4f}")
        # print("cls report", report)
        epochtime = time.time() - st
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        #log_file.write(f'Fold {fold+1}/{n_splits}, Epoch {epoch+1}/{num_epochs},  '
         #              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Training Cohen Kappa: {train_kappa:.4f}, Validation Cohen Kappa: {val_kappa:.4f} '
         #              f'Epoch Time: {epochtime:.2f}s\n')
    return train_loss_epoch, val_loss_epoch, train_accuracy_epoch, val_accuracy_epoch

def evaluate(model, classes, test_loader, criterion):
    """
    This function evaluate the model on the test set and returns the average loss, accuracy, kappa amd classification report.

    """   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

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
            
            # only for roc_curve, handle in calibration too
            probs = torch.softmax(outputs, dim=1)  # Get probabilities
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for the positive class

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    # this version confusion matrix only handles binary classification in case tertiary, it will return 9 values, yet to hnadle accordingly
    #cm = confusion_matrix(all_labels, all_preds).ravel()
    # Compute FPR, TPR, and ROC AUC score
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    # print("fpr, tpr, thresholds", fpr, tpr, thresholds)
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0, output_dict=True) # ['HC', 'HO', 'RE'],  
    return avg_loss, accuracy, kappa, report, [fpr.tolist(), tpr.tolist(), thresholds.tolist()]

def plot_epochs(fold, num_epochs_trans, train_loss_epoch, val_loss_epoch, train_accuracy_epoch, val_accuracy_epoch):
    # Generate epoch numbers for x-axis
    fig,(ax1,ax2) = plt.subplots(1,2)
    epochs = range(1, num_epochs_trans + 1)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_epoch, label='Train Loss', color='blue', linestyle='-')
    plt.plot(epochs, val_loss_epoch, label='Validation Loss', color='slateblue', linestyle='--')
    plt.plot(epochs, train_accuracy_epoch, label='Train Accuracy', color='red', linestyle='-')
    plt.plot(epochs, val_accuracy_epoch, label='Validation Accuracy', color='indianred', linestyle='--')

    # Labels and Title
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title(f"fold-{fold + 1} Training and Validation Metrics over Epochs")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_epochs_and_cls_report(fold, modelname, num_epochs_trans, train_loss_epoch, val_loss_epoch, train_accuracy_epoch, val_accuracy_epoch, report, save_dir=None, show=None):
    # Generate epoch numbers for x-axis
    epochs = range(1, num_epochs_trans + 1)
    
    # Create subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # Plotting losses on the first axis
    ax1.plot(epochs, train_loss_epoch, label='Train Loss', color='blue', linestyle='-')
    ax1.plot(epochs, val_loss_epoch, label='Validation Loss', color='slateblue', linestyle='--')
    ax1.plot(epochs, train_accuracy_epoch, label='Train Accuracy', color='red', linestyle='-')
    ax1.plot(epochs, val_accuracy_epoch, label='Validation Accuracy', color='indianred', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f"Fold-{fold + 1} Training and Testing")
    ax1.legend(loc='best')
    ax1.grid(True)
    

    # Plot heatmap on the second axis
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, ax=ax2, cmap='coolwarm')
    ax2.set_title("Classification Report on Val-set")
    
    # Save figure to directory if specified
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{modelname}_fold_{fold + 1}.png")
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Figure saved at {save_path}")

    # Show the plots
    plt.suptitle(f"Fold-{fold + 1} Training Metrics and Classification Report")
    plt.tight_layout()
    if show is not None:
        plt.show()

def show_batch(batch):
    """Plot images grid of single batch"""
    for images, labels in batch:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=8).permute(1,2,0))
        break
        
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

def save_results(class_str, file_path, fold_metrics, avg_metrics):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        f = open(file_path, "a")
    else:
        f = open(file_path, "w")
        # with open(file_path, 'a') as f:
    f.write(f"\n----Classification of {class_str}----\n")
    f.write("\nFold-Wise Results:\n")
    for i, fold in enumerate(zip(*fold_metrics.values())):
        f.write(f"Fold {i + 1}: {', '.join(f'{k}: {v:.2f}' for k, v in zip(fold_metrics.keys(), fold))}\n")
    f.write("\nAverage Metrics:\n")
    for metric, value in avg_metrics.items():
        f.write(f"{metric}: {value}\n")

def save_fold_dict(data_dict, filepath, subject):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        f  = open(filepath, "a")
    else:
        f= open(filepath, "w")
    f.write(f"\nSubject: {subject}\n")
    for metric, values in data_dict.items():
        data_dict[metric] = [f"{value:.2f}" for value in values]
    print(data_dict, file=f)

# Define a modified forward function that allows feature extraction too be used for t-SNE plotting
def extract_features(model, dataloader, layer="conv1"):
    model.eval()
    features = []
    labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, targets in dataloader:  # Assuming a DataLoader is used
            inputs = inputs.permute(0, 2, 1, 3, 4) # only for 3D-CNN
            inputs, targets = inputs.to(device), targets.to(device)
            feature_map = model(inputs, extract_features=True)  # Extract intermediate features
            features.append(feature_map.view(feature_map.shape[0], -1).cpu().numpy())  # Flatten
            labels.append(targets.cpu().numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)
    
    return features, labels


def plot_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="jet", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Extracted Features")
    plt.show()

def model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)