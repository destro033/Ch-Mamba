# C-Mamba and PatchTST

In this repo we used two multivariate time series forecasting models, the official codes can be found here:

1. [C-Mamba](https://github.com/zclzcl0223/CMamba)

2. [PatchTST](https://github.com/PatchTST/PatchTST)

These models are combined into a single file for the purpose of this work

## Installation

To use the models, you will need these libraries:
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `joblib`

If necessary, you can install them using pip:
```bash
pip install torch pandas numpy matplotlib scikit-learn joblib
```

## Training Procedure

In order to train the models, download as zip folder all the codes and store them in single folder.
For C-Mamba training you can change any parameters (if needed) from arguments.py
For PatchTST training you can change any parameters from arguments_patchtst.py. In these files hyperparameters of each model with their descriptions is provided.
After changes save these files and run train.py for C-Mamba or run train_patchtst.py for PatchTST.
Make sure that the dataset on here is downloaded as CSV in the same folder as the other codes. After training, the code will produce the files: mamba_best_model.pth and scaler_mamba.pkl for C-Mamba, and patchtst_best_model.pth and scaler_patchtst.pkl for PatchTST, these files contain the best weights and scaler for these models to be used in testing
.Also, the code will produce CSV file containing the train/val losses for each epoch. These files contain the best weights and scaler for these models to be used in testing

## Testing Procedure and Plotting

In order to test and plot the results of the models, download as zip folder all the codes and store them in single folder.
For C-Mamba, run test.py. After running the code cmamba_results.npz file will be produced, this file will be needed for plotting of the results.
For PatchTST, run test_patchtst.py. After running the code patchtst_results.npz file will be produced, this file will be needed for plotting of the results. 
Note that both of these files are necessary for plotting.

After getting both .npz files, run plot_results.py code to plot the results (make sure these files are called cmamba_results.npz and patchtst_results.npz and make sure that these files are in the same folder as plot_results.py)
After finishing, the metrics will appear and will be downloaded as PDFs as well.

