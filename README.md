# C-Mamba and PatchTST

In this repo we used two multivariate time series forecasting models, the official codes can be found here:
1. [C-Mamba](https://github.com/zclzcl0223/CMamba)
2. [PatchTST](https://github.com/PatchTST/PatchTST)

These models are combined into a single file for the purpose of this work C-Mamba is located in model.py, whereas PatchTST is located in model_patchtst.py

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

 **Note:** If you want to train the models, delete  `mamba_best_model.pth`, `scaler_mamba.pkl`, `patchtst_best_model.pth`, and `scaler_patchtst.pkl`  files (which already exist in this repo) beforehand, as they will be overwritten after training.

The training procedure is as follows:

1. Download the repository and extract all files into a single folder.

2. If necessary, edit the model hyperparameters:
   - For **C-Mamba**, edit `arguments.py`
   - For **PatchTST**, edit `arguments_patchtst.py`

3. Download the dataset from https://zenodo.org/records/15089283 and place the CSV file in the same folder as the code.

4. Run the training scripts:
   - For **C-Mamba**:
     ```bash
     python train.py
     ```
   - For **PatchTST**:
     ```bash
     python train_patchtst.py
     ```

5. After training, the following files will be generated:
   - `mamba_best_model.pth`, `scaler_mamba.pkl`
   - `patchtst_best_model.pth`, `scaler_patchtst.pkl`

   These files contain the trained model weights and corresponding data scalers.

6. CSV files containing training and validation losses for each epoch will also be generated.

## Testing Procedure and Plotting

In order to test and plot the results of the models, download as zip folder all the codes and store them in the same folder (in the zip there are also the pretrained models as mentioned in training procedure).
For C-Mamba, run test.py. After running the code cmamba_results.npz file will be produced, this file will be needed for the plotting of the results.
For PatchTST, run test_patchtst.py. After running the code patchtst_results.npz file will be produced, this file will be needed for plotting of the results as well. 
Both files will be automitically saved to your working folder
Note that both of these files are necessary for plotting.

After getting both .npz files, run plot_results.py code to plot the results (make sure these files are called cmamba_results.npz and patchtst_results.npz and make sure that these files are in the same folder as plot_results.py)
After finishing, the metrics will appear and will be downloaded as PDFs as well.

