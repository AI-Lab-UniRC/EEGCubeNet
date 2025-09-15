The proposed method EEGCubeNet was compared with [3D-EEGNet](https://pubmed.ncbi.nlm.nih.gov/37934650/).
The 3D-EEGNet was implemented from scratch, following the structure described in original [paper](https://pubmed.ncbi.nlm.nih.gov/37934650/).  
Both methods were assessed using a Leave-One-Subject-Out LOSO cross-validation strategy: each model was
trained on data from N-1 subjects and tested on the remaining subject. 

### How to run 3D-EEGNet model
```
1. You can compute the results of 3D-EEGNet by running following command in terminal
python eegnet3D.py
2. All the parameters are already set, however you can change them in main() method similar to EEGCubeNet experiments.
3. After running #eegnet3D.py you can read the results in the base directory created as described in the comments of #eegnet3D.py file.
4. To plot the comparative results reported in the EEGCubeNet paper, run following command in terminal
python plots.py
```
