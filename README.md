# BirdCLEF from (Kaggle)[https://www.kaggle.com/competitions/birdclef-2024]


# Bird Audio Classification

This project focuses on classifying 182 species of birds using audio recordings. The primary objective is to leverage various deep learning models and preprocessing techniques to achieve high classification accuracy. Specifically, the project employs ResNet and EfficientNet models trained on different types of spectrograms. The dataset was processed with and without silencing the quiet parts of the recordings, and multiple experiments were conducted to optimize the classification performance.

## Project Structure

The repository contains the following Jupyter notebooks, each documenting different stages and methodologies of the experiments:

1. **EfficientNet_default.ipynb**: Experiments using the EfficientNet model without silencing quiet parts of the audio.
2. **EfficientNet_silence.ipynb**: Experiments using the EfficientNet model with silenced quiet parts of the audio.
3. **ResNet_default.ipynb**: Experiments using the ResNet model without silencing quiet parts of the audio.
4. **ResNet_silence.ipynb**: Experiments using the ResNet model with silenced quiet parts of the audio.

## Spectrogram Types

For each model, five types of spectrograms were generated from the audio data for training and validation:
- **Mel Spectrogram**
- **Constant-Q Transform (CQT)**
- **Short-Time Fourier Transform (STFT)**
- **Spectral Contrast**
- **Chroma Feature**

Additionally, the Mel spectrograms, which yielded the best results, were used for further experiments involving:
- **Label Smoothing**
- **Data Augmentation**

## Experiment Overview

### Without Silence Suppression
- **EfficientNet_default.ipynb**
- **ResNet_default.ipynb**

These notebooks include:
- Training and validation on the five spectrogram types.
- Experiments with label smoothing and data augmentation using the Mel spectrogram.

### With Silence Suppression
- **EfficientNet_silence.ipynb**
- **ResNet_silence.ipynb**

These notebooks follow a similar experimental setup, but the audio data is preprocessed to silence the quiet parts before generating the spectrograms.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `librosa`
- `torch`
- `torchvision`
- `tqdm`
- `scikit-learn`
- `efficientnet_pytorch`

You can install the required libraries using:
```sh
pip install -r requirements.txt
