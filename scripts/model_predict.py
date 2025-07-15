import numpy as np
import os
import json
from scripts.utils import load_model_file, load_scaler_file


def load_feature_cols(ticker, model_type):
    feature_path = os.path.join(os.path.dirname(__file__), '../models', f'{ticker}_{model_type}_features.json')
    if os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Feature columns file not found for {ticker} {model_type}")


def predict_next_days(df, ticker, model_type='LSTM', days_ahead=1, seq_length=60):
    feature_cols = load_feature_cols(ticker, model_type)
    model = load_model_file(ticker, model_type)
    scaler = load_scaler_file(ticker, model_type)

    if model is None or scaler is None:
        raise ValueError("Model or scaler not found!")

    data = df[feature_cols].values
    data_scaled = scaler.transform(data)
    input_seq = data_scaled[-seq_length:].reshape(1, seq_length, 1)

    preds_scaled = []

    for _ in range(days_ahead):
        pred = model.predict(input_seq, verbose=0)[0, 0]
        preds_scaled.append(pred)
        next_input = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
        input_seq = next_input

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled)[:, 0]
    return preds
