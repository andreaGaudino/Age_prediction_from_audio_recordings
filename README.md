## Project Overview

This project focuses on building a machine learning model to estimate the age of a speaker based on acoustic features extracted from their voice recordings. The core approach involves extensive Feature Engineering from audio spectrograms, data preprocessing (including scaling and dimensionality reduction using PCA), and training a Ridge Regression model.

The complete methodology is detailed in the accompanying report (report.pdf).

## Methodology Highlights

### Feature Extraction
Leveraging the librosa library, we extracted advanced acoustic features to capture nuances in speech related to age, including:

- MFCCs (Mel-Frequency Cepstral Coefficients): Mean and standard deviation of 40 coefficients.

- Mel Spectrogram: Mean and standard deviation of 20 log-power Mel bands.

- Chroma Features, Spectral Rolloff, and Spectral Bandwidth.

### Preprocessing

- Outlier Detection: Removed recordings with features falling significantly outside the standard deviation bounds ($t=\mu\pm3\cdot\sigma$) to improve model stability.

- Feature Cleaning: Categorical features (gender) were One-Hot Encoded. The ethnicity feature was removed due to high cardinality and distribution mismatch between datasets.

- Scaling and PCA: Applied Standard Scaling (which performed best) and Principal Component Analysis (PCA) to normalize data and reduce feature dimensionality, respectively, thereby mitigating multicollinearity and improving computational efficiency.

### Model Selection

Ridge Regressor: Chosen for its effectiveness in handling linear regression with a large number of potentially correlated features (the Principal Components). It uses L2 regularization to prevent overfitting.

### Results

The model achieved a public RMSE score of 9.264 on the evaluation data, demonstrating a reasonably precise prediction of speaker age.

## Repository Structure

- `feature_extraction.ipynb`: Contains all Python code for data cleaning, preprocessing, and the custom function used to extract acoustic features (MFCCs, Mel-spectrogram, Chroma, etc.) from the audio files.

- `project.ipynb`: Contains the core machine learning pipeline: data loading, train/validation/test splitting, implementation of different scaling methods (column norm, standard scaling), PCA application, GridSearchCV for hyperparameter tuning, and final Ridge Regression training and evaluation.

- `report.pdf`: The detailed report outlining the problem, the theoretical background, the full methodology, and quantitative results.

- `datasets/`: Directory (expected to contain the development.csv, evaluation.csv, and the audio files used for feature extraction).

## How to Run the Project

### Prerequisites

You need a Python environment with the following packages installed:

*`pip install pandas numpy scikit-learn librosa nbimporter matplotlib`*


**(Note: nbimporter is required to import functions between the Jupyter Notebooks.)**

### Step-by-Step Execution

- Setup Data: Ensure your dataset files (development.csv, evaluation.csv, and the corresponding audio files) are correctly placed in the ./datasets/ directory as referenced in the notebooks.

- Run Feature Extraction:

  - Execute the feature_extraction.ipynb notebook completely. This performs all data cleaning, feature engineering, One-Hot Encoding, and outlier detection.

  - This notebook saves the processed dataframes to the ./data_completi/ directory (e.g., df_tot.csv, dev.csv, eval.csv).

- Run ML Pipeline:

  - Open and run the project.ipynb notebook. This notebook:

    - Loads the pre-processed data.

    - Implements the scaling and PCA transformations.

    - Performs cross-validated GridSearchCV to tune the alpha parameter for Ridge Regression.

    - Trains the final models and generates the prediction files for the evaluation set.
