import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')

def save_model(model, ticker, model_type):
    model_path = os.path.join(MODEL_DIR, f'{ticker}_{model_type}_model.h5')
    model.save(model_path)
    print(f'Saved model: {model_path}')

def load_model_file(ticker, model_type):
    model_path = os.path.join(MODEL_DIR, f'{ticker}_{model_type}_model.h5')
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None

def save_scaler(scaler, ticker, model_type):
    scaler_path = os.path.join(MODEL_DIR, f'{ticker}_{model_type}_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f'Saved scaler: {scaler_path}')

def load_scaler_file(ticker, model_type):
    scaler_path = os.path.join(MODEL_DIR, f'{ticker}_{model_type}_scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        return None

def evaluate_predictions(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # Untuk MAPE, hindari pembagian nol dengan filter actual != 0
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if np.any(mask) else np.nan
    return mae, rmse, mape
