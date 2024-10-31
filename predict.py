#!/usr/bin/env python3
"""
predict.py

This script loads a trained LSTM model and uses it to make predictions on new multivariate time series data.
Usage:
    python predict.py <path_to_input_file>

The input file can be in CSV, Excel, or Parquet format. The script preprocesses the data, makes predictions,
and saves the results in the same directory with an added suffix.
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings("ignore")

seed = 99
np.random.seed(42)
tf.random.set_seed(42)


def load_model(model_path):
    """Load the trained LSTM model."""
    return tf.keras.models.load_model(model_path)


def load_data(file_path):
    """Load data from CSV, Excel, or Parquet file."""
    _, file_ext = os.path.splitext(file_path)
    if file_ext.lower() == '.csv':
        return pd.read_csv(file_path, sep=';', decimal=',')
    elif file_ext.lower() in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif file_ext.lower() == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, Excel, or Parquet file.")


def preprocess_data(df, lower_bound, upper_bound, n_timesteps):
    """
    Preprocess the input data:
    - Replace missing values
    - Drop unnecessary columns
    - Create timestamp
    - Normalize the data
    - Create sequences
    """
    # Replace -200 with NaN and interpolate
    df = df.replace(to_replace=-200, value=np.nan).interpolate()

    # Drop columns with all NaNs if any
    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=['Date'])

    # Create Timestamp if not present
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%d/%m/%Y %H.%M.%S")
        df = df.drop(['Date', 'Time'], axis=1)

    # Select relevant columns (the same as training)
    required_columns = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'NO2(GT)']
    df = df[['Timestamp'] + required_columns]
    df = df.sort_values('Timestamp', ascending=True)

    # Normalize the data using provided bounds
    df = df.iloc[-n_timesteps:, 1:]
    array_norm = np.expand_dims(((df - lower_bound) / (upper_bound - lower_bound)).values, 0)

    print(array_norm.shape)

    return array_norm  # shape=(1, n_timesteps, 6)


def save_predictions(predictions, output_path, file_format, lower_bound, upper_bound):
    """Save the predictions to a new file with the same format as input. First denormalization."""
    predictions = predictions * (upper_bound[-1] - lower_bound[-1]) + lower_bound[-1]
    predictions = pd.Series(predictions.squeeze(), name='Predicted_NO2(GT)')
    if file_format.lower() == '.csv':
        predictions.to_csv(output_path, index=False, sep=';')
    elif file_format.lower() in ['.xls', '.xlsx']:
        predictions.to_excel(output_path, index=False)
    elif file_format.lower() == '.parquet':
        predictions.to_parquet(output_path, index=False)
    else:
        raise ValueError("Unsupported file format for saving.")


def main():
    """Main function to execute the prediction pipeline."""
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_input_file>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        sys.exit(1)

    # Define parameters
    model_path = os.path.join('models', 'lstm_model.keras')
    n_timesteps = 500

    # Load the model
    model = load_model(model_path)

    # Load the data
    df = load_data(input_path)

    # Define normalization bounds (should be the same as training)
    lower_bound = joblib.load(os.path.join('data', 'interim', 'bounds', 'lower_bound.pkl'))
    upper_bound = joblib.load(os.path.join('data', 'interim', 'bounds', 'upper_bound.pkl'))

    # Preprocess the data
    X = preprocess_data(df, lower_bound, upper_bound, n_timesteps)

    # Make predictions
    predictions = model.predict(X)

    # Define output path
    directory, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_predictions{ext}"
    output_path = os.path.join(directory, output_filename)

    # Save the predictions
    save_predictions(predictions, output_path, ext, lower_bound, upper_bound)

    print(f"Predictions saved to '{output_path}'.")


if __name__ == "__main__":
    main()
