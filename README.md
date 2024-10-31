# ASG-TestDataScience-2

## Project Overview

This repository contains a project focused on multivariate time series regression using non-financial data without seasonality. The goal is to predict air quality metrics, specifically the concentration of NO2, using a Long Short-Term Memory (LSTM) model.

## Repository Structure

### **data**

- **external**: New data to be processed in production.
- **interim**: Holds intermediate data that has been transformed during preprocessing.
- **processed**: Final datasets ready for modeling.
- **raw**: Original data dumps that remain unchanged.

### **models**

- Stores trained models.

### **notebooks**

#### `main.ipynb`

- This notebook is used for exploratory data analysis and contains code snippets to understand and visualize the dataset. It may also include steps for model training or evaluation.

### **scripts**

#### `predict.py`

- **Functionality**: This script loads a trained LSTM model to make predictions on new multivariate time series data.
- **Key Functions**:
  - `load_model(model_path)`: Loads the trained LSTM model from a specified path.
  - `load_data(file_path)`: Loads input data from CSV, Excel, or Parquet files.
  - `preprocess_data(df, lower_bound, upper_bound, n_timesteps)`: Prepares the data by handling missing values, normalizing, and creating sequences.
  - `save_predictions(predictions, output_path, file_format)`: Saves the predictions in the same format as the input file.
  - `main()`: Manages the prediction pipeline including loading the model and data, preprocessing, predicting, and saving results.


## Setup Instructions

1. **Environment Setup**: Ensure you have all dependencies installed using `requirements.txt` with the command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Running Predictions**:
   - Prepare your input file in CSV, Excel, or Parquet format.
   - Run the prediction script:
     ```bash
     python predict.py <path_to_input_file>
     ```

3. **Model Training**: Use Jupyter notebooks in the `notebooks` directory to train models or perform further analysis.