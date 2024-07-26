<!-- # 2-Stage-OpAmp-Analysis -->
<!-- Comparative Analysis of Machine Learning Models for Aspect Ratio Estimation of a Two-Stage Operational Amplifier -->

# Aspect Ratio Estimation of a Two-Stage Operational Amplifier

This repository contains a Streamlit web application that estimates the aspect ratios of a two-stage operational amplifier using various machine learning models. The application allows users to input specific parameters and select a model to predict the aspect ratios.

## Features

- **Interactive UI**: User-friendly interface to input parameters and select models.
- **Multiple Models**: Provides predictions using different regression models including Linear Regression, Gaussian Process Regression, SVR, Decision Tree, KNN, Random Forest, and a Neural Network.
- **Visualization**: Displays predictions and aspect ratios for the selected model.

## Getting Started

### Prerequisites

- Python 3.x
- Streamlit
- Keras
- Scikit-learn
- Numpy
- Pandas
- Matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/2-stage-opamp-aspect-ratio.git
   cd 2-stage-opamp-aspect-ratio
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the dataset `2STAGEOPAMP_DATASET.csv` in the same directory.

4. Ensure you have the trained models `model.h5` and `gaussian_model.pkl` in the same directory.

### Running the App

Run the Streamlit app using the following command:
```bash
streamlit run main.py
```

This will start the Streamlit server, and you can interact with the app in your web browser.

## Usage

1. **Input Features**:
   - DC Gain
   - Unity Gain Frequency (ft)
   - 3-dB Frequency (f3)
   - Common Mode Voltage (Vcm)
   - Power Dissipation (Pdiss)

2. **Select a Model**:
   - Linear Regression Model
   - Gaussian Regression Model
   - SVR
   - Decision Tree Regressor
   - KNN
   - Random Forest Regressor
   - Neural Network (Best)

3. **Get Predictions**: Click the 'Calculate' button to get the predicted aspect ratios for the given input features.

## Code Overview

### `main.py`

- **Imports**: Necessary libraries including Streamlit, Numpy, Pandas, Scikit-learn, and Keras.
- **Data Loading**: Loads the dataset `2STAGEOPAMP_DATASET.csv` and preprocesses it.
- **Model Loading**: Loads the pre-trained models for prediction.
- **Model Functions**: Defines functions for each machine learning model to predict aspect ratios.
- **Streamlit UI**: Creates the sidebar and main panel for user input and model selection.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
