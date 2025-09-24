## Processing EEG Volumetric Data
- EEG Lab is recommended to process the .mat files
- Read the dataset from ![Data](https://github.com/AI-Lab-UniRC/EEGCubeNet/tree/main/Data) folder.
The flowchart of EEG volumetric data processing is folloiwng:
!(https://github.com/AI-Lab-UniRC/EEGCubeNet/blob/main/EEG%20Video%20PreProcessing/eegvideoprocessing.PNG)

It starts from EEG acquisition, pre-processing and dataset creation. The diagram illustrates the data acquisition
paradigm. EEG segments of 1 second preceding the onset of motion referred to as pre-motion EEG segments are extracted, labeled (as HC, HO, or RE), and stored in a dataset. These EEG signals undergo spatial filtering using the Laplacian method, followed by time-frequency analysis via the Continuous Wavelet Transform (CWT). The resulting time-frequency representations are structured into volumes organized by channel, frequency, and time. These volumes are then labeled accordingly and stored for further analysis.

#### Quick recape
```
1. Each EEG segment was spatially filtered to reduce volume conduction and to reduce signal correlations between adjacent electrodes.
2. Spatially filtered EEG are projected into time-frequency (TF) domain using Continuous Wavelet Transform (CWT).
3. Since TF was performed for each channel, hence it resultsed 59 maps
4. Such maps are stacked to form a 3D matrix with dimensions: channel x frequency x time (spatial, spectral, and temporal).
5. Motor planning and preparation related frequncy bands i.e., movement-related cortical potentials (MRCPs < 5 HZ):
sensorymotor rythoms -> 13-15 Hz
B-band -> 13-40 Hz
Analysed on -> 0.5-40 Hz
6. A set of 59 pseudo frequencies was generated using #scal2freq in #MATLAB2024b
7. In the end, 3D maps sized 59x59x512 were created and then downsampled to 59x59x128 for faster computation.
```
