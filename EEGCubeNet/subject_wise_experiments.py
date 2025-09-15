import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

from utils import set_seed
from utils import split_data, extract_metrics, compute_avg_std, save_results
from read_eeg_datasets import ReadVideoDataset
from eegcubenet import Deep3DCNN



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
        val_loss, val_accuracy, val_kappa, report = evaluate(model, classes, val_loader, criterion)

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
    return train_loss_epoch, val_loss_epoch, train_accuracy_epoch, val_accuracy_epoch, model

def evaluate(model, classes, test_loader, criterion):
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


def main():

    # For subject-by-subject experiments and results are stored in corresponding directories.
    subjects = ['S01'] # ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14']  #  List of subjects
    base_data_dir = '/home/neuronas/Motion_Planning_Upper_Limb_Dataset/WEnergy_movies_59x59x128'
    base_save_dir = '/home/sufflin/Documents/IJNS-GitHub/Subject-Wise'
    classes = ['HO', 'RE']
    class_str = 'HO-RE'
    modelname = "HORE"

    n_splits = 10
    batch_size = 8
    num_epochs = 50
    lr = 0.001
    no_classes = 2

    for subject in subjects:
        set_seed(42)
        print(f"Processing Subject: {subject}")
        data_dir = f"{base_data_dir}/{subject}"
        save_dir = f"{base_save_dir}/{subject}"
        os.makedirs(save_dir, exist_ok=True)
        results_file = f"{save_dir}/{modelname}/results_foldwise_3dcnn.txt"

        ### Reads the single subject data from the data_dir
        dataset = ReadVideoDataset(data_dir, classes, frames=128, height=59, width=59)


        train_data, val_data, test_data = split_data(dataset)
        merged_train_test = train_data + test_data


        ### K-fold train-test and validation 
        kfold = KFold(n_splits=n_splits, shuffle=True) 
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

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
            
            print(f"Fold-{fold+1}/{n_splits}")
            # Create data loaders for training and validation sets
            train_subset = torch.utils.data.Subset(merged_train_test, train_idx) # merged_train_test
            val_subset = torch.utils.data.Subset(merged_train_test, val_idx) # dataset


            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)


            # Check if GPU is available and set the device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = Deep3DCNN(no_classes).to(device) 

            # for param in model.parameters():
            #     param.requires_grad = True # should be True when we need parameters to be learned

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
            scaler = GradScaler()


            train_loss, val_loss, train_acc, val_acc, cnn_model = train_and_test(model, classes, scaler, scheduler, optimizer, criterion, train_loader, val_loader, num_epochs)
            
        
            #model_filename = 'model_S' + str(subject_id) + '.pth'
            #torch.save(model.state_dict(), model_filename)
            

            val_loader_video = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            avg_loss, acc, kappa, report = evaluate(model, classes, val_loader_video, criterion)
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}, Test Kappa: {kappa:.4f}")
            print(report)


            ## To save the best fold model for calibration
            if acc > best_accuracy:
                best_accuracy = acc  # Update the best accuracy
                # Save the current model
                model_filename = "Best_3DCNN.pth"
                # ### Save the model
                torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
            else:
                print("Current fold did not beat the best accuracy, skipping model save.")
            
            
            # plot_epochs_and_cls_report(fold, modelname, num_epochs, train_loss, val_loss, train_acc, val_acc, report, save_dir=save_dir, show=None)

            metrics = extract_metrics(report)

            fold_metrics['accuracy'].append(metrics['accuracy'])
            fold_metrics['precision'].append(metrics['precision_macro'])
            fold_metrics['recall'].append(metrics['recall_macro'])
            fold_metrics['f1'].append(metrics['f1_macro'])

        # Compute average and std for subject
        avg_metrics = compute_avg_std(fold_metrics)

        # Save fold-wise and overall metrics
        save_results(class_str, results_file, fold_metrics, avg_metrics)

        # model_filename = "Finetuned_3DCNN-All.pth"
        # # ### Save the model
        # torch.save(model.state_dict(), os.path.join(save_dir, model_filename))

     
if __name__ == "__main__":
    print("Running Subject-wise experiments.....")
    main()