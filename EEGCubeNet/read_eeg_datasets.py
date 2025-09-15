import torch
import mat73
import os
import re
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class ReadMatricesDataset(Dataset):
    def __init__(self, mat_file_path, classes, str1, str2, mapping, transform=None):
        """
        Args:
            mat_file_path (str): Path to the .mat file containing the processed data.
            transform (callable, optional): Optional transform to apply to data.
            data_str is to handle dynamically the already saved variables like combinedData for all subjects and allData for single subject
        """
        self.mat_file_path = mat_file_path
        self.transform = transform
        self.labels = []


        data_dict = mat73.loadmat(self.mat_file_path)
        print(data_dict.keys())
        
        self.data = data_dict[str1][:].astype('float32')  # [59, 128, 59, 180], 
        # print("actual shape of all data", self.data.shape)
        self.alllabels = data_dict[str2][:] #.astype('int64')  # [180]
        # print("actual shape of all labels", self.alllabels)

        # mapping = {'HC':0, 'HO':1, 'RE':2} # for three-way
        # mapping = {'HC':0, 'RE':1} # for two0way
        # mapping = {'HO':0, 'RE':1} # for two0way
        # mapping = {'HC':0, 'HO':1} # for two0way
        for label in self.alllabels:
            # print(mapping[label])
            self.labels.append(mapping[label])
        # print("debug to see labels", self.labels)
        
        # Convert the shape to match PyTorch convention: [N, C, H, W]
        self.data = self.data.transpose(3, 1, 2, 0)  # [180, 128, 59, 59]
        # print("after shape of all data", self.data.shape)
        # Add a channel dimension:
        self.data = np.expand_dims(self.data, axis=2) # [180, 128, 1, 59, 59]
        print("shape of matrices all data", self.data.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the data sample and its corresponding label
        sample = self.data[idx]  # Shape: [128, 1, 59, 59]
        label = self.labels[idx]  # Integer label

        # Convert the sample to a PyTorch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        #label = torch.tensor(label, dtype=torch.long)
        #label = label.squeeze()

        # Apply any optional transformations
        if self.transform:
            sample = self.transform(sample)


        return sample, torch.tensor(label, dtype=torch.long)


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

        # Load and process the video (grayscale already)
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

            # # print("Frame shape in load video:", frame.shape)
            # plt.imshow(frame)
            # plt.axis('off')  # Hide axes for a cleaner display
            # plt.show()
            # plt.pause(0.5)  # Pause to view each frame; adjust as needed
            # plt.clf()  # Clear figure after each frame
            # Resize to x by x 
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
        # print("tensor shape and values", video_tensor[0])  

        return torch.tensor(video_tensor)

"""ReadVideoDataset class reads the all folders from the base directory and combines the video data of all subjects."""


class ReadVideoDatasetAllSubjects(Dataset):
    """
    This class reads videos from the specified base directory and performs preprocessing 
    to make sure each video has the correct dimensions. It normalizes the video tensors by 255 
    to get tensor values in the range [0, 1].
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



class ReadVideoDataset4Trials(Dataset):
    """
    This class reads the videos and perform same operations as ReadVideoDataset, however, it 
    allows to read specific videos based on trials. We can set how many trials we need to read from each subject.
    """
    def __init__(self, data_dir, classes, frames=128, height=206, width=206, start_trial=1, end_trial=60):   # but motion planning dataset, max 60 trials per subject
        self.data_dir = data_dir
        self.classes = classes
        self.frames = frames
        self.height = height
        self.width = width
        self.video_files = []
        self.labels = []
        self.start_trial = start_trial
        self.end_trial = end_trial

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
                    # Extract the trial number using regex (assuming the format Movie_trial_<number>.avi)
                    match = re.search(r'Movie_trial_(\d+)', file)
                    if match:
                        trial_number = int(match.group(1))
                        # Check if the trial number is within the given range
                        if self.start_trial <= trial_number <= self.end_trial:
                            video_path = os.path.join(class_dir, file)
                            self.video_files.append(video_path)
                            self.labels.append(label)
                            # print(f"Added file: {video_path}")  # Debug print
                        # else:
                            # print(f"Skipping file: {file} (Trial {trial_number} out of range)")
                    else:
                        print(f"Skipping file: {file} (No valid trial number found)")

        print(f"Total video files loaded: {len(self.video_files)}")  # Debug print


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label = self.labels[idx]

        # Load and process the video (grayscale already)
        video_tensor = self.load_video(video_file)

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)

        frames_list = []
        frame_count = 0
        while frame_count < self.frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize 
            frame = cv2.resize(frame, (self.width, self.height))

            # Add a channel dimension (grayscale video so it has only 1 channel)
            frame = np.expand_dims(frame, axis=0)

            # If the frame has an extra channel (like 3 for RGB), remove it
            if frame.shape[-1] == 3:
                # print("yes channels are 3")
                frame = frame[..., 0]  # Keep only one chanel (grayscale)

            frames_list.append(frame)
            frame_count += 1

        cap.release()

        # Pad video with empty frames if it has fewer than required frames
        while len(frames_list) < self.frames:
            frames_list.append(np.zeros((1, self.height, self.width))) # 3 is channel, but its 1

        # Convert list to numpy array and normalize to [0, 1]
        video_tensor = np.array(frames_list, dtype=np.float32) / 255.0

        return torch.tensor(video_tensor)


class GrayscaleVideoDataset(Dataset):
    def __init__(self, data_dir, classes, frames=128, height=59, width=59, start_trial=1, end_trial=60):
        self.data_dir = data_dir
        self.classes = classes
        self.frames = frames
        self.height = height
        self.width = width
        self.video_files = []
        self.labels = []
        self.start_trial = start_trial
        self.end_trial = end_trial

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
                    # Extract the trial number using regex (assuming the format Movie_trial_<number>.avi)
                    match = re.search(r'Movie_trial_(\d+)', file)
                    if match:
                        trial_number = int(match.group(1))
                        # Check if the trial number is within the given range
                        if self.start_trial <= trial_number <= self.end_trial:
                            video_path = os.path.join(class_dir, file)
                            self.video_files.append(video_path)
                            self.labels.append(label)
                            # print(f"Added file: {video_path}")  # Debug print
                        # else:
                            # print(f"Skipping file: {file} (Trial {trial_number} out of range)")
                    else:
                        print(f"Skipping file: {file} (No valid trial number found)")

        print(f"Total video files loaded: {len(self.video_files)}")  # Debug print


    """ def __init__(self, data_dir, classes, frames=128, height=206, width=206):
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

        print(f"Total video files loaded: {len(self.video_files)}")  # Debug print  """

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label = self.labels[idx]

        # Load and process the video (grayscale already)
        video_tensor = self.load_video(video_file)

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)

        frames_list = []
        frame_count = 0
        while frame_count < self.frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to 224x224 (instead of 128x128)
            frame = cv2.resize(frame, (self.width, self.height))

            # Add a channel dimension (grayscale video so it has only 1 channel)
            frame = np.expand_dims(frame, axis=0)

            # If the frame has an extra channel (like 3 for RGB), remove it
            if frame.shape[-1] == 3:
                # print("yes channels are 3")
                frame = frame[..., 0]  # Keep only one channel (grayscale)

            frames_list.append(frame)
            frame_count += 1

        cap.release()

        # Pad video with empty frames if it has fewer than required frames
        while len(frames_list) < self.frames:
            frames_list.append(np.zeros((1, self.height, self.width))) # 3 is channel, but its 1

        # Convert list to numpy array and normalize to [0, 1]
        video_tensor = np.array(frames_list, dtype=np.float32) / 255.0

        return torch.tensor(video_tensor)

""" The following function preprocess the dataset and splits into train-test and validation parts. """
class VideoDatasetSplit(Dataset):
    def __init__(self, data_dir, classes, frames=128, height=206, width=206, start_trial=1, end_trial=600, validation_split=0.15, test_split=0.15):
        self.data_dir = data_dir
        self.classes = classes
        self.frames = frames
        self.height = height
        self.width = width
        self.video_files = []
        self.labels = []
        self.start_trial = start_trial
        self.end_trial = end_trial

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
                    # Extract the trial number using regex (assuming the format Movie_trial_<number>.avi)
                    match = re.search(r'Movie_trial_(\d+)', file)
                    if match:
                        trial_number = int(match.group(1))
                        # Check if the trial number is within the given range
                        if self.start_trial <= trial_number <= self.end_trial:
                            video_path = os.path.join(class_dir, file)
                            self.video_files.append(video_path)
                            self.labels.append(label)
                        else:
                            print(f"Skipping file: {file} (Trial {trial_number} out of range)")
                    else:
                        print(f"Skipping file: {file} (No valid trial number found)")

        print(f"Total video files loaded: {len(self.video_files)}")  # Debug print

        # Now split the dataset into train, test, and validation sets
        # First split into train + validation, and test
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            self.video_files, self.labels, test_size=test_split, random_state=42, stratify=self.labels)

        # Then split train and validation
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels, test_size=validation_split/(1 - test_split), random_state=42, stratify=train_val_labels)

        # Assign to self
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

        print(f"Train files: {len(self.train_files)}, Validation files: {len(self.val_files)}, Test files: {len(self.test_files)}")

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        video_path = self.train_files[idx]
        label = self.train_labels[idx]
        # Load and process the video (grayscale already)
        video_tensor = self.load_video(video_path)

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def load_video(self, video_file):
        cap = cv2.VideoCapture(video_file)

        frames_list = []
        frame_count = 0
        while frame_count < self.frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to 224x224 (instead of 128x128)
            frame = cv2.resize(frame, (self.width, self.height))

            # Add a channel dimension (grayscale video so it has only 1 channel)
            frame = np.expand_dims(frame, axis=0)

            # If the frame has an extra channel (like 3 for RGB), remove it
            if frame.shape[-1] == 3:
                # print("yes channels are 3")
                frame = frame[..., 0]  # Keep only one channel (grayscale)

            frames_list.append(frame)
            frame_count += 1

        cap.release()

        # Pad video with empty frames if it has fewer than required frames
        while len(frames_list) < self.frames:
            frames_list.append(np.zeros((1, self.height, self.width))) # 3 is channel, but its 1

        # Convert list to numpy array and normalize to [0, 1]
        video_tensor = np.array(frames_list, dtype=np.float32) / 255.0

        return torch.tensor(video_tensor)



class FrameDataset(Dataset):
    """ 
    The following module reads videos and extracts frames and stores frames data with their related labels 
    """
    def __init__(self, video_dataset):
        self.video_dataset = video_dataset  # Original dataset of videos
        self.frame_labels = []  # List to store each frame with corresponding label
        
        # Extract frames and labels from each video
        for video, label in self.video_dataset:
            num_frames = video.shape[0]  # 128 frames per video
            for i in range(num_frames):
                self.frame_labels.append((video[i], label))  # Add each frame with label
    
    def __len__(self):
        return len(self.frame_labels)
    
    def __getitem__(self, idx):
        frame, label = self.frame_labels[idx]
        return frame, label


def extract_and_shuffle_frames(video_dataset):
    """
    Extract frames from the videos in the dataset and shuffle the frames and their labels.
    
    Args:
    video_dataset: A dataset where each entry contains a (video, label) pair.
    
    Returns:
    shuffled_frames: A list of tuples where each tuple is (frame, label) with frames shuffled.
    """
    frame_labels = []

    # Extract frames and labels from each video in the dataset
    for video, label in video_dataset:
        num_frames = video.shape[0]  # Number of frames in the video
        for i in range(num_frames):
            frame_labels.append((video[i], label))  # Append each frame with its corresponding label

    # Shuffle the frame-label pairs
    random.shuffle(frame_labels)
    
    return frame_labels