This repository contains the source code and data used in [XYZ Paper], accepted for publishing in [International Journal of Neural Systems](https://www.worldscientific.com/worldscinet/ijns?srsltid=AfmBOoonImUsPjx0i9bH8102pIJpjb6tnnCIwOlrvlum3LwmQt_ipqXb)."
It presents the technical details with source code to reproduce the state of the art results for the underlying problem in BCI.  

## Introduction
- EEGCubeNet along with 3D xAI-OSA is an interpretable classification system for video EEG related to Motor Imagery in BCI.
- A novel deep learning-based EEG decoding model is proposed to project EEG signals by taking into account simoultaneously the spatial, spectral, and temporal characteristics of EEG signals.
- State-of-the-art performances were achieved with no need of solving the inverse problem to reconstruct EEG cortical sources, which is computationally expensive and requires a head model.
- The proposed model #EEGCubeNet was extensively validated to leverage the knowledge acquired during training on multiple subjects (global training) to the adaptation to the unseen final subject (subject-wise fine-tuning). 
- The proposed approach enabled the achievement of state-of-the-art performance on individual subjects while significantly reducing the training time.

## Dataset 
The EEG data used in this project is sourced from [Ofner et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182578), [BNCI Horizon 2020](https://bnci-horizon-2020.eu/database/data-sets).
The dataset includes:
- 15 Participants
- Channels: 61 
- Epoched Data: Sampled at 512 Hz
- Band Pass Filtered: 0.01 - 200 Hz
- 3 Classes: Hand Open (HO), Hand Close (HC), and Resting (RE)

## EEGCubeNet Model and Fine-tuning strategy
A 3D Convolutional Neural Networks (3D-CNNs) is designed with four 3D convolutional layers, each followed by Batch Normalization, ReLU activation, Max Pooling, and Conv layers are foolowed by 3 fully connected layers. The input to the model is a 3D volume of size $128 \times 59 \times 59$ (representing channels vs. frequency) with 128 frames.
The architecture of EEGCubeNet is folloiwng:
![EEGCubeNet](https://github.com/M-Suffian/EEGCubeNet/blob/main/EEGCubeNet.PNG)
The layer-wise fine-tuning is illustrated using different colors. The shaded area with different colors represents the progressive fine-tuning process, starting from the last layer and extending to the entire model.
At each step of fine-tuning, we analyzed its impact on the performance of EEGCubeNet. 

## Training Setup
- Kfolds = 10
- Optimizer: Adam
- Batch size: 8
- Epochs: 50
- Learning Rate: 0.001
- Loss Function: Cross Entropy
- Scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

## 3D-xAI-OSA Interpretability Method
A novel 3D xAI-OSA approach that explains, simultaneously, the EEG process from spatial, spectral, and temporal perspectives.
The architecture of 3D xAI-OSA is presented below:
![3D-xAI-OSA](https://github.com/M-Suffian/EEGCubeNet/blob/main/xAI_3D_OSA.PNG)
EEG volumetric data channels x frequency x time is occluded using masks of varying sizes (a sliding 3D mask) and processed by a trained EEGCubeNet model. The upper section demonstrate the original input to EEGCubeNet and its classification score is computed, the bottom section illustrates masked input volume processed with EEGCubeNet to compute the classification score. The relevance is computed for both original and masked input based on multiple factors, leading to a saliency map representing the relevance in all dimensions.

> [!TIP]
> Clone the repository and extract the corresponding zip into your desired local directory. Open the terminal and cd to extracted folder. Then, run the commands as prescribed in the Quick Start!. Good Luck!

## Quick Start!!

The experiments are conducted for inter and intra-subject (leave-one-subject-out and subject-wise).
All the source code files for both experiments are provided in #EEGCubeNet.

To run the subject-wise experiments:
```
Consider following details in  #subject_wise_experiments.py file:
1. The default script run the experiments for a single subject S01, if you want to run the experiments for all subjects and to analyse the results of EEGCubeNet model, then, open the comment for multiple subjects from source code line #114 in subject_wise_experiments.py file.
2. You have to set the base data directory from line 115, this dataset is the processed video data in .mat or .mp4 or .avi format obtained after processing volumetric EEG data. If you have not yet processed the data then follow these instructions in [EEG Video PreProcessing]. 
3. To run experiments for a specific tasks such as HC vs. RE or HO vs. RE classification, then, specifiy your desired string and classes in lines 117-119.
4. Further, you can customise the hyperparameters from lines 121-125.
5. After running the following command in terminal, you can see the results in the parent directory you already chosen, with a new sub-directory 'Subject-Wise/S01/HCRE/foldwise_results.txt' containing fold-wise training and testing results for HCRE task on S01 data.

python subject_wise_experiments.py
```

To run the LOSO-CV experiments:
```
Consider following details in #loso_cv_experiments.py file:
1. Since in this experiment it is supposed to perform finetuning on the left-out subject, you must specifiy or choose from the level of finetuning indicated with specific string such as Conv1, Conv2...FC3, etc. at line 37. After this, the script will create a folder in the parent directory to store the corresponding results for this level of model fine-tuing.
2. The default script run the experiments for all subjects, using LOSO approach, N-1 subjects for training and Nth subject for fine-tuning. You can customise it at line 46.
3. To run experiments for a specific tasks such as HC vs. RE or HO vs. RE classification, then, specifiy your desired string and classes at line 50 and line 63.
4. You have to set the data directory at line 49 from where to read the data, and save directory at line 51 where to save the results of specific task.
5. Further, you can customise the hyperparameters from lines 57-61.
6. If you want to perform few-shot lerning then you have to set the target directory at line 62 from where to read target data, and set target classes at line 63. If no few shot learning then keep the both parameters same as the data directory and classes at line 49-50, respectively.
7. After running the following command, you can see the results in the parent directory you already chosen, with a new sub-directory 'Conv1/AllSubjects-vs-Calibrated-S01/HCRE/log.txt' containing fold-wise training, test, and fine-tuning results for HCRE task on S01 data when trained on 13 subjects and fine-tuned on S01.

python loso_cv_experiments.py
```

To plot the t-SNE for any level of fine-tuning run the following file and follow instructions in the file:
```
python t_sne.py
```

To get interpretable and explainable results of the decoding model for spatial-spectral and temporal insights, do the following:
```
# The code is provided for multiple kind of analysis in #3d_xai_osa.py file with comments.
# Then, run the following command in terminal:

python 3d_xai_osa.py
```

### Requirements
```
You need to install the required libraries by installing the requirements.txt file.

pip install -r /path/to/requirements.txt
```

> [!NOTE]
> To run succesfully the experiments and reporoduce the results reported in the corresponding paper to this repository, you must make sure to have installed Python 3.11 and Pytorch framework.
> You must run these experiments on the GPU-based machine to compare the time reported in our paper. The dataset is a video-based (volumetric) data, the scripts for the EEGCubeNEt and for other operations are written in Pytorch friendly python and you have to respect these constarints for a successful reproducibility.


> [!IMPORTANT]
> The experiments described in Quick Start! only work with processed EEG data, to learn how the EEG data is processed as Video EEG, then visit [EEG Video PreProcessing](https://github.com/M-Suffian/EEGCubeNet/tree/main/EGG%20Video%20PreProcessing)

> [!CAUTION]
> The experiments were implemented using PyTorch an open-source deep learning framework, running on a workstation equipped with Ubuntu.
> Model training was performed on an NVIDIA RTX 4000 Ada Generation installed on a processor of Intel Xeon(R) CPU @2.30GHz and RAM of 125 GB.

## Citing
```
Suffian, Muhammad et al., "Explainable 3D-Deep Learning Model for EEG Decoding in Brain Computer Interface Applications."
International Journal of Neural Systems X.X (2025): XYZ. 
```

Bibtex
```
@article{suffian3DEEGCubeNet,
  title={Explainable 3D-Deep Learning Model for EEG Decoding in Brain Computer Interface Applications},
  author={Suffian, Muhammad and Ieracitano, Cosimo and Morabito, Francesco Carlo and Mammone, Nadia},
  journal={International Journal of Neural Systems},
  volume={X},
  number={X},
  pages={XYZ},
  year={2025},
  publisher={World Scientific}
}
```

#### Cite following as baseline papers
1- [A Few-Shot Transfer Learning Approach for Motion Intention Decoding from Electroencephalographic Signals](https://pubmed.ncbi.nlm.nih.gov/38073546/)
```
@article{mammone2023few,
  title={A Few-Shot Transfer Learning Approach for Motion Intention Decoding from Electroencephalographic Signals.},
  author={Mammone, Nadia and Ieracitano, Cosimo and Spataro, Rossella and Guger, Christoph and Cho, Woosang and Morabito, Francesco Carlo},
  journal={International Journal of Neural Systems},
  volume={34},
  number={2},
  pages={2350068 (19 pages)},
  year={2024}
}
```
2- [A hybrid-domain deep learning-based BCI for discriminating hand motion planning from EEG sources](https://pubmed.ncbi.nlm.nih.gov/34376121/)
```
@article{ieracitano2021hybrid,
  title={A hybrid-domain deep learning-based BCI for discriminating hand motion planning from \textsc{EEG} sources},
  author={Ieracitano, Cosimo and Morabito, Francesco Carlo and Hussain, Amir and Mammone, Nadia},
  journal={International journal of neural systems},
  volume={31},
  number={09},
  pages={2150038},
  year={2021},
  publisher={World Scientific}
}
```
3- [A novel Explainable Machine Learning Approach for \textsc{EEG}-based Brain-Computer Interface Systems](https://www.springerprofessional.de/en/a-novel-explainable-machine-learning-approach-for-eeg-based-brai/18937780)
```
@article{ieracitano2021_NCA,
  title={A novel Explainable Machine Learning Approach for \textsc{EEG}-based Brain-Computer Interface Systems},
 author={Ieracitano, Cosimo and Mammone, Nadia and Hussain, Amir and Morabito, Francesco Carlo},
  journal={Neural Computing and Applications},
  volume={34},
  number={14},
  pages={11347--11360},
  year={2022},
  publisher={Springer}
}
```
4- [Decoding Motor Preparation Through a Deep Learning Approach Based on EEG Time-Frequency Maps](https://link.springer.com/chapter/10.1007/978-3-031-24801-6_12)
```
@inproceedings{mammone2022decoding,
  title={Decoding Motor Preparation Through a Deep Learning Approach Based on \textsc{EEG} Time-Frequency Maps},
  author={Mammone, Nadia and Ieracitano, Cosimo and Spataro, Rossella and Guger, Christoph and Cho, Woosang and Morabito, Francesco C},
  booktitle={International Conference on Applied Intelligence and Informatics},
  pages={159--173},
  year={2022},
  organization={Springer}
}
```
5- [A deep CNN approach to decode motor preparation of upper limbs from time–frequency maps of EEG signals at source level](https://pubmed.ncbi.nlm.nih.gov/32045838/)
```
@article{mammone_2020_BCI,
  title={A deep \textsc{CNN} approach to decode motor preparation of upper limbs from time–frequency maps of \textsc{EEG} signals at source level},
  author={Mammone, Nadia and Ieracitano, Cosimo and Morabito, Francesco C.},
  journal={Neural Networks},
  volume={124},
  number={},
  pages={357-372},
  year={2020},
  publisher={Elsevier}
}
```


## Licence
```
CC-BY 4.0 
```
