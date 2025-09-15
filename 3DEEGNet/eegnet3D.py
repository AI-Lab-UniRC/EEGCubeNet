import torch
import os
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold



from utils import train_and_test, split_data, evaluate, extract_metrics, compute_avg_std
from utils import set_seed
from utils import EarlyStopping, ReadVideoDataset, ReadVideoDatasetAllSubjects


class EEGNet3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet3DCNN, self).__init__()
        
        # First Convolutional block
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=59, kernel_size=(1, 1, 13), stride=(1, 1, 1), padding=(1, 1, 1)) #16 channels in orig
        self.bn1 = nn.BatchNorm3d(16)

        
        # Depth-wise convolutional block
        self.depthconv1 = nn.Conv3d(in_channels=59, out_channels=128, kernel_size=(59, 1, 1), stride=(1, 1, 1), padding=(1, 1, 1))
        self.depth1bn2 = nn.BatchNorm3d(32)

        self.depthconv2 = nn.Conv3d(in_channels=59, out_channels=128, kernel_size=(1, 59, 1), stride=(1, 1, 1), padding=(1, 1, 1))
        self.depth2bn2 = nn.BatchNorm3d(32)
        self.depthpool2 = nn.MaxPool3d(kernel_size=(1, 1, 4), stride=(1, 1, 1))
        
        # Separabale Convolutional block
        self.sepconv1 = nn.Conv3d(in_channels=59, out_channels=128, kernel_size=(1, 1, 4), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 1, 12), stride=(1, 1, 1))

        # Fully connected layers
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        # x: [batch_size, channels, frames, height, width]
        print("shape of x", x.shape)
        x = self.bn1(self.conv1(x))
        x = self.depth1bn2(self.depthconv1(x))
        x = self.depthpool2(self.depth2bn2(self.depthconv2(x)))  
        x = self.pool3(self.bn3(self.sepconv1(x)))  

        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        return x
    

class EEGNet3DCNN1(nn.Module):
    def __init__(self, num_classes=2):
        super(EEGNet3DCNN1, self).__init__()

        # First 3D Convolutional Block: Temporal Filtering
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=16,
            kernel_size=(1, 1, 13),  # temporal filters
            stride=(1, 1, 1), padding=(0, 0, 6)  # keep time dimension same
        )
        self.bn1 = nn.BatchNorm3d(16)

        # Block 2-1: Depthwise Spatial Filtering (Vertical, front-to-back)
        self.depthwise_vert = nn.Conv3d(
            in_channels=16, out_channels=16,  # depthwise: one filter per channel
            kernel_size=(59, 1, 1),  # vertical across channels
            stride=(1, 1, 1), padding=(0, 0, 0),
            groups=16,  # depthwise
        )
        self.bn2 = nn.BatchNorm3d(16)

        # Block 2-2: Depthwise Spatial Filtering (Horizontal, ear-to-ear)
        self.depthwise_horiz = nn.Conv3d(
            in_channels=16, out_channels=16,
            kernel_size=(1, 59, 1),
            stride=(1, 1, 1), padding=(0, 0, 0),
            groups=16  # depthwise
        )
        self.bn3 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 4), stride=(1, 1, 4))  # reduce time by 4

        # Block 3: Separable Convolution
        self.sepconv = nn.Conv3d(
            in_channels=16,  # match output from previous layer
            out_channels=128,
            kernel_size=(1, 1, 4),
            stride=(1, 1, 1),
            padding=(0, 0, 1)
        )
        self.bn4 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 1, 4), stride=(1, 1, 4))  # further reduce time

        # Fully connected layer
        self.fc = nn.Linear(26880, num_classes)  # assumes time is reduced to 8

    def forward(self, x):
        # Input: [batch, 1, 59, 59, 128]
        # x = x.unsqueeze(1)  # add channel dim if not present
        # # If your data is [128, 59, 59, 128]
        # x = x.permute(0, 1, 2, 3, 4)

        x = self.conv1(x)  # [B, 16, 59, 59, 128]
        x = self.bn1(x)
        x = F.elu(x)

        x = self.depthwise_vert(x)  # [B, 59, 1, 59, 128]
        x = self.bn2(x)
        x = F.elu(x)

        x = self.depthwise_horiz(x)  # [B, 59, 1, 1, 128]
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)  # [B, 59, 1, 1, 32]

        x = self.sepconv(x)  # [B, 128, 1, 1, 30]
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool3(x)  # [B, 128, 1, 1, 8]

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)  # [B, 2]
        return x

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


def main():


    crossSubject_metrics = {'subject':[], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'kappa':[]}

    # Define dataset and dataloader with 59*59
    subjects =  ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14'] #'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 
    for subject in subjects:
        print(f"Processing Subject: {subject}")
        data_dir = '/home/neuronas/Motion_Planning_Upper_Limb_Dataset/WEnergy_movies_59x59x128'  # Path to the folder containing the video files
        classes = ['HO', 'RE']  # Subfolders representing class labels (e.g., '0', '1')
        save_dir = f"/home/sufflin/Documents/MotionPlanning/IJNS-2025/EEGNet/AllSubjects-vs-Calibrated-{subject}/HORE"
        results_file = f"{save_dir}/log_list.txt"
        class_str = 'All' 
        class_str_calib = 'Calibrated'
        set_seed(42)
        batch_size = 8
        num_epochs = 15
        n_splits = 10
        lr = 0.001
        no_classes = 2
        target_dir = f"/home/neuronas/Motion_Planning_Upper_Limb_Dataset/WEnergy_movies_59x59x128/{subject}"
        classes_target = ['HO', 'RE']
        # model_path = f"/home/sufflin/Documents/MotionPlanning/IJNS-2025/EEGNet/AllSubjects-vs-Calibrated-{subject}/HORE"
        model_path = f"/home/sufflin/Documents/MotionPlanning/IJNS-2025/EEGNet/CS-FoldModels/HORE"

        dataset = ReadVideoDatasetAllSubjects(data_dir, classes, subject=subject, frames=128, height=59, width=59) # For Motion Planning data the shape is 59x59 and each class has 60 trials.
        

        # For vlidating the model on test set
        train_data, test_data, val_data = split_data(dataset) 
        merged_train_test = train_data + test_data


        ### K-fold train-test and validation 
        kfold = KFold(n_splits=n_splits, shuffle=True) 

        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'kappa':[]}
        fold_confmatrix = {'tn': [], 'fp': [], 'fn': [], 'tp': []}
        fold_roc = {'fpr':[], 'tpr':[], 'thresholds':[]}

        train_loss_fold = []
        val_loss_fold = []
        train_accuracy_fold = []
        val_accuracy_fold = []
        train_kappa_fold = []
        val_kappa_fold = []
        best_accuracy = 0.0

        # K-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(merged_train_test)): #merged_train_test or dataset
            
            print(f"Fold-{fold+1}/{n_splits}")
            # Create data loaders for training and validation sets
            train_subset = torch.utils.data.Subset(merged_train_test, train_idx) # merged_train_test
            val_subset = torch.utils.data.Subset(merged_train_test, val_idx) # dataset


            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)


            # Check if GPU is available and set the device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = EEGNet3DCNN1(no_classes).to(device) # Initializing an instance of EEGNet class
            # model.apply(reinitialize_weights)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
            scaler = GradScaler()


            # call here the train and evaluare fnction
            ts = time.time()
            train_loss, val_loss, train_acc, val_acc = train_and_test(model, classes, scaler, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs)
            te = time.time()
            print("Training time for 1 fold is:", te-ts)


            val_loader_video = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            avg_loss, acc, kappa, report, roc = evaluate(model, classes, val_loader_video, criterion)
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}, Test Kappa: {kappa:.4f}")

            ## To save the best fold model for calibration
            if acc > best_accuracy:
                best_accuracy = acc  # Update the best accuracy
                # Save the current model
                model_filename = f"LOSO_EEGNet-All-{subject}.pth"
                # ### Save the model
                torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            else:
                print("Current fold did not beat the best accuracy, skipping model save.")
        
            metrics = extract_metrics(report)

            fold_metrics['accuracy'].append(metrics['accuracy'])
            fold_metrics['precision'].append(metrics['precision_macro'])
            fold_metrics['recall'].append(metrics['recall_macro'])
            fold_metrics['f1'].append(metrics['f1_macro'])
            fold_metrics['kappa'].append(kappa)
            # tn, fp, fn, tp = cm.ravel()
            #fold_confmatrix['tn'].append(cm[0])
            #fold_confmatrix['fp'].append(cm[1])
            #fold_confmatrix['fn'].append(cm[2])
            #fold_confmatrix['tp'].append(cm[3])
            # roc curve items
            #fold_roc['fpr'].append(roc[0])
            #fold_roc['tpr'].append(roc[1])
            #fold_roc['thresholds'].append(roc[2])

        # Compute average and std for subject
        avg_metrics = compute_avg_std(fold_metrics)
        # df_fold_metrics = pd.DataFrame(fold_metrics)
        # df_fold_metrics.to_latex(f"{save_dir}/table.tex", index=False) # f"{subject}.tex", index=False, escape=False, caption="Metrics Table", label="tab:example", column_format="|c|c|c|", longtable=True, float_format="%.2f", bold_rows=True)


        # Save fold-wise and overall metrics
        save_results(class_str, results_file, fold_metrics, avg_metrics)

        #save_fold_dict(fold_metrics, results_file, subject)
        #save_fold_dict(fold_confmatrix, results_file, subject)
        # pd.DataFrame(fold_roc).to_csv("ROC.csv")
        crossSubject_metrics_fold = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'kappa':[]}
        time_folds = []

        for fold in range(10):

            # model_filename = f"LOSO_EEGNet-All-{subject}.pth"
            model_filename = f"EEGNet3D-{subject}-{fold}.pth"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            trained_model = EEGNet3DCNN1(no_classes).to(device) # EEGNet
            trained_model.load_state_dict(torch.load(os.path.join(model_path, model_filename)))
            
            print(f" ----- Model testing on Target Subject: {subject}-------")
            ### Repeating the testing process for 10 times and each time shuffling the data to get the std like 10 fold
            validation_data_target = ReadVideoDataset(target_dir, classes_target, frames=128, height=59, width=59) # when to test on leave-one-out subject data
            val_loader_video = DataLoader(validation_data_target, batch_size=batch_size, shuffle=False)
            for name, param in trained_model.named_parameters():
                param.requires_grad = False
            criterion = nn.CrossEntropyLoss()
            ts = time.time()
            avg_loss, acc, kappa, report, roc = evaluate(trained_model, classes_target, val_loader_video, criterion)
            tf = time.time()
            tot = tf-ts
            time_folds.append(tot)
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}, Test Kappa: {kappa:.4f}")
            print("Time for this fold to inference",tot) 
            metrics = extract_metrics(report)
            crossSubject_metrics_fold['accuracy'].append(metrics['accuracy'])
            crossSubject_metrics_fold['precision'].append(metrics['precision_macro'])
            crossSubject_metrics_fold['recall'].append(metrics['recall_macro'])
            crossSubject_metrics_fold['f1'].append(metrics['f1_macro'])
            crossSubject_metrics_fold['kappa'].append(kappa)
        print("time for all folds", np.mean(time_folds))
        print(crossSubject_metrics_fold)

        crossSubject_metrics['subject'].append(subject)
        crossSubject_metrics['accuracy'].append(np.std(crossSubject_metrics_fold['accuracy']))
        crossSubject_metrics['precision'].append(np.std(crossSubject_metrics_fold['precision']))
        crossSubject_metrics['recall'].append(np.std(crossSubject_metrics_fold['recall']))
        crossSubject_metrics['f1'].append(np.std(crossSubject_metrics_fold['f1']))
        crossSubject_metrics['kappa'].append(np.std(crossSubject_metrics_fold['kappa']))
    print(crossSubject_metrics)
    #df_crosssub_metrics = pd.DataFrame(crossSubject_metrics)
    #df_crosssub_metrics.to_latex("/home/sufflin/Documents/MotionPlanning/IJNS-2025/EEGNet/HORE_CS_leftout_metrics_repeat.tex", index=False) # escape=False, caption="Metrics Table", label="tab:example", column_format="|c|c|c|", longtable=True, float_format="%.2f", bold_rows=True)



if __name__ == "__main__":
    print("Running cnn_frame.py")
    main()