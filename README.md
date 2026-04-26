# C-Mamba and PatchTST
In this repo we used two multivariate time series forecasting models, the official codes can be found here:

1. [C-Mamba](https://github.com/zclzcl0223/CMamba)

2. [PatchTST](https://github.com/PatchTST/PatchTST)

These models are combined into a single file for the purpose of this work

The full training code is on FULLTRAINING.py

The full testing code is on FULLTESTING.py

You can also build these codes manually as explained below.

## Training Procedure
In order to build the code for training the model, put the following codes in the following order:
1. Imports.py
2. (Colab) Upload dataset
3. Preprocessing.py 
4. Model.py
5. Arguments.py
6. Training.py
7. (Optional) Epochs.py

## Testing Procedure
If you are planning to use the pretrained model on the dataset mentioned put the following codes in order:
1. Imports.py
2. (Colab) Upload files.py here upload the args.pth, scaler.pkl, weights.pth
