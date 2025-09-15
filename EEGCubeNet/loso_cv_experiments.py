import torch
import os
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset, random_split, Subset, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import KFold

from utils import set_seed
from utils import train_and_test, evaluate, split_data, split_data_traintest
from utils import EarlyStopping, model_parameters
from utils import extract_metrics, compute_avg_std, save_fold_dict, save_results
from read_eeg_datasets import ReadVideoDataset, ReadVideoDatasetAllSubjects
from eegcubenet import Deep3DCNN



def main():

    ###  This script need to run when to perofrm the Leave-One-Out-Subject (LOSO) training , finetuning and testing. 
    ## The following script reads N-1 subjects data and utilise it for K-Fold training fientuning of the model.
    # The best model of k-folds is utilised for the fine-tuning on the left-out subject.

    
    ### The fine'tune level string is important as it will create corresponding directory to save the results subject by subject.
    ## The following are advised levels: [Conv1, Conv2, Conv3, Conv4, FC1, FC2, FC3]
    finetune_level = "FC3" # [Conv1, Conv2, Conv3, Conv4, FC1, FC2, FC3] it means the layers to fine tune. FC3 means from FC3 to last layer of model gonna be finetuned.
    task = 'HORE'
    if task == 'HCRE':
        classA = 'HC'
        classB = 'RE'
    else:
        classA = 'HO'
        classB = 'RE'

    subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14']
    for subject in subjects:
        print(f"Processing Subject: {subject}")
        data_dir = '/home/neuronas/Motion_Planning_Upper_Limb_Dataset/WEnergy_movies_59x59x128'  # Path to the folder containing the video EEG files already processed and ready to use for training.
        classes = ['HC', 'RE']  # Subfolders of the main data_directory representing class labels (e.g., '0', '1') to read class-wise data 
        save_dir = f"/home/sufflin/Documents/IJNS-GitHub/{finetune_level}/AllSubjects-vs-Calibrated-{subject}/{task}"  # create folder directory to save the results for each case
        os.makedirs(save_dir, exist_ok=True)
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
        classes_target = ['HO', 'RE']  #  these can be different if you want to perform few shot learning
        model_path = f"/home/sufflin/Documents/IJNS-GitHub/{finetune_level}/AllSubjects-vs-Calibrated-{subject}/{task}" # reading trained model from any other folder is same.
        os.makedirs(model_path, exist_ok=True)
        
        dataset = ReadVideoDatasetAllSubjects(data_dir, classes, subject=subject, frames=128, height=59, width=59) # Gonna read N-1 subjects data instead of target subject 


        # dataset = GrayscaleVideoDataset(data_dir, classes, frames=128, height=59, width=59, start_trial=1, end_trial=60)

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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = Deep3DCNN(no_classes).to(device) # Initializing an instance of learning model
            # model.apply(reinitialize_weights)

            # for param in model.parameters():
            #     param.requires_grad = True # should be True when we need parameters to be learned

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
            scaler = GradScaler()


            ## call here the train and evaluare fnction
            ts = time.time()
            train_loss, val_loss, train_acc, val_acc = train_and_test(model, classes, scaler, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs)
            te = time.time()
            print("Training time for 1 fold is:", te-ts)
            
            #model_filename = 'model_S' + str(subject_id) + '.pth'
            #torch.save(model.state_dict(), model_filename)
            

            val_loader_video = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            avg_loss, acc, kappa, report, roc = evaluate(model, classes, val_loader_video, criterion)
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}, Test Kappa: {kappa:.4f}")

            ## To save the best fold model for calibration
            if acc > best_accuracy:
                best_accuracy = acc  # Update the best accuracy
                # Save the current model
                model_filename = "LOSO_3DCNN-All.pth"
                # ### Save the model
                torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            else:
                print("Current fold did not beat the best accuracy, skipping model save.")
            
            # ## To plot t-SNE plots for the feature maps from the last layer
            # model_tsne = Deep3DCNN(no_classes, extract_features=True).to(device)
            # model_tsne.load_state_dict(torch.load(os.path.join(model_path, 'Finetuned_3DCNN-All.pth')))
            # features_tsne, labels_tsne = extract_features(model_tsne, val_loader_video, layer="conv1")
            # plot_tsne(features_tsne, labels_tsne)

            # # To plot the evaluationmetrics results fold-by-fold in save_dir
            plot_str = f"All"
            # plot_epochs_and_cls_report(fold, plot_str, num_epochs, train_loss, val_loss, train_acc, val_acc, report, save_dir=save_dir, show=None)

        #     metrics = extract_metrics(report)

        #     fold_metrics['accuracy'].append(metrics['accuracy'])
        #     fold_metrics['precision'].append(metrics['precision_macro'])
        #     fold_metrics['recall'].append(metrics['recall_macro'])
        #     fold_metrics['f1'].append(metrics['f1_macro'])
        #     fold_metrics['kappa'].append(kappa)
        #     # tn, fp, fn, tp = cm.ravel()
        #     # fold_confmatrix['tn'].append(cm[0])
        #     # fold_confmatrix['fp'].append(cm[1])
        #     # fold_confmatrix['fn'].append(cm[2])
        #     # fold_confmatrix['tp'].append(cm[3])
        #     # roc curve items
        #     # fold_roc['fpr'].append(roc[0])
        #     # fold_roc['tpr'].append(roc[1])
        #     # fold_roc['thresholds'].append(roc[2])

        # # Compute average and std for subject
        # avg_metrics = compute_avg_std(fold_metrics)
        # # df_fold_metrics = pd.DataFrame(fold_metrics)
        # # df_fold_metrics.to_latex(f"{save_dir}/table.tex", index=False) # f"{subject}.tex", index=False, escape=False, caption="Metrics Table", label="tab:example", column_format="|c|c|c|", longtable=True, float_format="%.2f", bold_rows=True)


        # # Save fold-wise and overall metrics
        # save_results(class_str, results_file, fold_metrics, avg_metrics)

        # save_fold_dict(fold_metrics, results_file, subject)
        # # save_fold_dict(fold_confmatrix, results_file, subject)
        # # pd.DataFrame(fold_roc).to_csv("ROC.csv")


        # # model_filename = "Finetuned_3DCNN-All.pth"
        # # # ### Save the model
        # # torch.save(model.state_dict(), os.path.join(save_dir, model_filename))

        

        print(f" ----- Model Calibration/fine-tuning on Target Subject: {subject}-------")

        validation_data_target = ReadVideoDataset(target_dir, classes_target, frames=128, height=59, width=59) # when to test on leave-one-out subject data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_data_target, val_data_target = split_data_traintest(validation_data_target) # #  Splitting the validation part of the N-1 subjects data
        merged_train_test = train_data_target 

        ### K-fold train-test and validation 
        kfold = KFold(n_splits=n_splits, shuffle=True) 
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'kappa':[]}
        fold_confmatrix = {'tn': [], 'fp': [], 'fn': [], 'tp': []}
        fold_roc = {'fpr':[], 'tpr':[], 'thresholds':[]}

        # Initialize lists to store metrics for each fold
        train_loss_fold = []
        val_loss_fold = []
        train_accuracy_fold = []
        val_accuracy_fold = []
        train_kappa_fold = []
        val_kappa_fold = []
        best_accuracy = 0.0
       
        # K-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(merged_train_test)): #merged_train_test or dataset
            print(f"Fold: {fold + 1}")
            
            # Create data loaders for training and validation sets
            train_subset = torch.utils.data.Subset(merged_train_test, train_idx) # merged_train_test
            val_subset = torch.utils.data.Subset(merged_train_test, val_idx) # dataset

            train_loader_target = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader_target = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            trained_model = Deep3DCNN(no_classes)
            trained_model.load_state_dict(torch.load(os.path.join(model_path, 'LOSO_3DCNN-All.pth')))

            # # The correct use of model.named_parameters() is for fine-tuning
            # for name, param in trained_model.named_parameters():
            #     print(f"Layer: {name}, Shape: {param.shape}")

            trained_model = trained_model.to(device)
            print(" Before fine-tuning the all parameters of the model:", model_parameters(trained_model))
            # from torchsummary import summary
            # summary(trained_model, input_size=(1, 128, 59, 59))

            # First of all, make sure all layer parameters are freezed and then select parameters of chosen layer
            for name, param in trained_model.named_parameters():
                param.requires_grad = False

            if finetune_level == "FC3":
                for name, param in trained_model.named_parameters():
                    if name.startswith(('fc3')):
                        param.requires_grad = True
            elif finetune_level == "FC2":
                for name, param in trained_model.named_parameters():
                    if name.startswith(('fc2', 'fc3')):
                        param.requires_grad = True
            elif finetune_level == "FC1":
                for name, param in trained_model.named_parameters():
                    if name.startswith(('fc1', 'fc2', 'fc3')):
                        param.requires_grad = True
            elif finetune_level == "Conv4":
                for name, param in trained_model.named_parameters():
                    if name.startswith(('conv4', 'fc1', 'fc2', 'fc3')):
                        param.requires_grad = True
            elif finetune_level == "Conv3":
                for name, param in trained_model.named_parameters():
                    if name.startswith(('conv3', 'conv4', 'fc1', 'fc2', 'fc3')):
                        param.requires_grad = True
            elif finetune_level == "Conv2":
                for name, param in trained_model.named_parameters():
                    if name.startswith(('conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'fc3')):
                        param.requires_grad = True
            elif finetune_level == "Conv2":
                for name, param in trained_model.named_parameters():
                    if name.startswith(('conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'fc3')):
                        param.requires_grad = True
            elif finetune_level == "Conv1": 
                for name, param in trained_model.named_parameters():
                    param.requires_grad = True
            else:
                print("No finetuning level is selected, please use layer name to fientune.")

            print(" After fine-tuning the all parameters of the model:", model_parameters(trained_model))
            
            trained_model = trained_model.to(device)

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, trained_model.parameters()), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()  # Assuming classification task
            scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
            scaler = GradScaler()

            ts = time.time()
            train_loss_epoch, val_loss_epoch, train_accuracy_epoch, val_accuracy_epoch = train_and_test(trained_model, classes_target, scaler, scheduler, optimizer, criterion, train_loader_target, val_loader_target, num_epochs)
            te = time.time()
            print("Calibration time for 1 fold is:", te-ts)

            val_loader_video = DataLoader(val_data_target, batch_size=batch_size, shuffle=False)
            avg_loss, acc, kappa, report, roc = evaluate(trained_model, classes_target, val_loader_video, criterion)
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}, Test Kappa: {kappa:.4f}")
            # print(report)
            ## To save the best fold model as calibrated model
            if acc > best_accuracy:
                best_accuracy = acc  # Update the best accuracy
                # Save the current model
                model_filename = "Finetuned_3DCNN.pth"
                # ### Save the model
                torch.save(trained_model.state_dict(), os.path.join(save_dir, model_filename))
                print(f"New best calibrated model saved with accuracy: {best_accuracy:.4f}")
            else:
                print("Current fold did not beat the best accuracy, skipping model save.")
            
            # plot_epochs_and_cls_report(fold, 'calibrtaed', num_epochs, train_loss_epoch, val_loss_epoch, train_accuracy_epoch, val_accuracy_epoch, report, save_dir=save_dir, show=None)
            
            metrics = extract_metrics(report)
            fold_metrics['accuracy'].append(metrics['accuracy'])
            fold_metrics['precision'].append(metrics['precision_macro'])
            fold_metrics['recall'].append(metrics['recall_macro'])
            fold_metrics['f1'].append(metrics['f1_macro'])
            fold_metrics['kappa'].append(kappa)
            # tn, fp, fn, tp = cm.ravel()
            # fold_confmatrix['tn'].append(cm[0])
            # fold_confmatrix['fp'].append(cm[1])
            # fold_confmatrix['fn'].append(cm[2])
            # fold_confmatrix['tp'].append(cm[3])
            # roc curve elements
            fold_roc['fpr'].append(roc[0])
            fold_roc['tpr'].append(roc[1])
            fold_roc['thresholds'].append(roc[2])

        # Compute average and std for subject
        avg_metrics = compute_avg_std(fold_metrics)

        # df_fold_metrics = pd.DataFrame(fold_metrics)
        # df_fold_metrics.to_latex(f"{save_dir}/{subject}_metrics.tex", index=False) # escape=False, caption="Metrics Table", label="tab:example", column_format="|c|c|c|", longtable=True, float_format="%.2f", bold_rows=True)

        # Save fold-wise and overall metrics
        save_results(class_str_calib, results_file, fold_metrics, avg_metrics)
        save_fold_dict(fold_metrics, results_file, subject)
        # # save_fold_dict(fold_confmatrix, results_file, subject)
        # fold_roc_mean = {}
        # for key, values in fold_roc.items():
        #     max_len = max(map(len, values))  # Find the longest list
        #     padded_values = [lst + [np.nan] * (max_len - len(lst)) for lst in values]  # Pad shorter lists with NaN
        #     fold_roc_mean[key] = np.nanmean(padded_values, axis=0).tolist()
        # df = pd.DataFrame(fold_roc_mean)
        # df.to_csv(f"{save_dir}/fold_roc_mean.csv", index=False)
        # save df to latex table

if __name__ == "__main__":
    print("Running LOSO CV experiments....")
    main()