# C-Mamba and PatchTST

In this repo we used two multivariate time series forecasting models for UAV Trajectory Prediction, the official codes can be found here:
1. [C-Mamba](https://github.com/zclzcl0223/CMamba)
2. [PatchTST](https://github.com/PatchTST/PatchTST)
   
For PatchTST, we adopted the model from [here](https://github.com/thuml/Time-Series-Library)

These models are combined into a single file for the purpose of this work.
C-Mamba is located in `model.py`, whereas PatchTST is located in `model_patchtst.py`

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

The training procedure is as follows:

1. Download the repository and extract all files into a single folder. **Note:** Delete  `mamba_best_model.pth`, `scaler_mamba.pkl`, `patchtst_best_model.pth`, and `scaler_patchtst.pkl`  files (which already exist in this repo) beforehand, as they will be overwritten after training.

2. If necessary, edit the model hyperparameters:
   - For **C-Mamba**, edit `arguments.py`
   - For **PatchTST**, edit `arguments_patchtst.py` and then save the files

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

The testing procedure is as follows:

1. Download the repository and extract all files into a single folder. The folder already includes the pretrained models (`mamba_best_model.pth`, `scaler_mamba.pkl`, `patchtst_best_model.pth`, and `scaler_patchtst.pkl`). If you trained the model from the beginning make sure they  have the mentioned names.
2. Download the dataset from https://zenodo.org/records/15089283 and place the CSV file in the same folder as the code.
   
4. Run testing for each model:
   - For **C-Mamba**:
     ```bash
     python test.py
     ```
     This will produce `cmamba_results.npz`

   - For **PatchTST**:
     ```bash
     python test_patchtst.py
     ```
     This will produce `patchtst_results.npz`

5. Both `.npz` files will be automatically saved in the working directory.

6. Run the plotting script:
   ```bash
   python plot_results.py

**Note:** Both .npz files are needed before running `plot_results.py`  
After running the code, pdf files will be created with the plots.

The results from this work are in `results` folder.
