# TFT-based PFU Network

This repository contains the code used to evaluate a 3D/2D DenseNet-style model for PFU detection from time-lapse `.mat` data. The main workflow is:

1. Run inference on example data with [test_detection_8frame.py](test_detection_8frame.py)
2. Post-process the per-patch scores in MATLAB with [ResultsRead_TxT.m](ResultsRead_TxT.m)

The Example Data can be found in: https://drive.google.com/drive/u/0/folders/19mrmmP8wwMerNz4dNcx9a4shqT66f4DN

## Repository Layout

- [Dataset.py](Dataset.py) - dataset loader used by training
- [network_v3.py](network_v3.py) - model definition
- [train_detection_8frame.py](train_detection_8frame.py) - training script
- [test_detection_8frame.py](test_detection_8frame.py) - inference script
- [ResultsRead_TxT.m](ResultsRead_TxT.m) - MATLAB post-processing and statistics script
- [Example_Data/](Example_Data/) - sample input and output structure
- [model_detection_epoch277.pth](model_detection_epoch277.pth) - example trained checkpoint

## Requirements

Python packages used by the scripts include:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `h5py`
- `tqdm`
- `configobj`
- `tensorboardX`
- `scikit-learn`
- `matplotlib`

You will also need MATLAB if you want to run [ResultsRead_TxT.m](ResultsRead_TxT.m).

## Quick Start

### 1. Install Python dependencies

Install the packages above in your preferred environment. For example:

```bash
pip install torch torchvision numpy scipy h5py tqdm configobj tensorboardX scikit-learn matplotlib
```

### 2. Review the hardcoded paths

The scripts were written for a Windows network drive and include absolute paths that you will likely want to edit before running:

- [train_detection_8frame.py](train_detection_8frame.py) points to the training and validation `.mat` folders
- [train_detection_8frame.py](train_detection_8frame.py) also sets the model save directory and log file path
- [test_detection_8frame.py](test_detection_8frame.py) loads [Example_Data/Input/diff_stack_multiparas.mat](Example_Data/Input/diff_stack_multiparas.mat) by default and writes results into [Example_Data/Output/](Example_Data/Output/)

### 3. Run inference on example data

Run:

```bash
python test_detection_8frame.py
```

Inference details:

- Loads the checkpoint from `model_detection_epoch277.pth`
- Reads `Example_Data/Input/diff_stack_multiparas.mat`
- Writes one score file per time point into `Example_Data/Output/log_*/`
- Produces patch-level class probabilities and TIFF visualizations

### 4. Post-process in MATLAB

Open [ResultsRead_TxT.m](ResultsRead_TxT.m) in MATLAB and run it after inference. The script:

- Reads the `result_time*_epoch277.txt` files
- Builds PFU masks and connected components
- Computes PFU count, total area, area per PFU, and growth statistics
- Saves summary data to `final_statics_test.mat`

## Data Format Notes

The training loader in [Dataset.py](Dataset.py) expects each `.mat` file to contain an `input` array. The script crops the central `50 x 50` region and normalizes each channel independently.

The test script expects a `diff_stack_multiparas` variable inside `diff_stack_multiparas.mat` and converts it into the shape used by the model.

## Model Notes

- The model is defined in [network_v3.py](network_v3.py)
- The architecture mixes 3D convolutions for temporal processing and 2D convolutions after the temporal depth is reduced
- The example checkpoint [model_detection_epoch277.pth](model_detection_epoch277.pth) matches the inference script configuration

## Tips

- If you change the number of input channels, frames, or crop size, make sure the dataset loader, model initialization, and checkpoint all stay consistent.
- The scripts currently assume a CUDA-capable environment when available and use `cuda:1` by default.
- If you run into path issues on a different machine, search for `Y:/` and `./Example_Data/` in the scripts and update them to local paths.
- Any questions? Welcome to contact liyuzhu@ucla.edu.
