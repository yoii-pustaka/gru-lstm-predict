import numpy as np
from scripts.utils import load_model_file, load_scaler_file

def predict_next_days(df, ticker, model_type='LSTM', days_ahead=1, seq_length=60):
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    close_idx = feature_cols.index('Close')

    model = load_model_file(ticker, model_type)
    scaler = load_scaler_file(ticker, model_type)
    if model is None or scaler is None:
        raise ValueError("Model or scaler not found!")

    data = df[feature_cols].values
    data_scaled = scaler.transform(data)
    input_seq = data_scaled[-seq_length:].reshape(1, seq_length, len(feature_cols))

    preds_scaled = []
    for _ in range(days_ahead):
        pred = model.predict(input_seq)[0, 0]
        preds_scaled.append(pred)

        last_features = input_seq[0, -1, :].copy()
        last_features[close_idx] = pred

        input_seq = np.append(input_seq[:, 1:, :], [[last_features]], axis=1)

    # Buat array untuk inverse transform, hanya kolom Close diisi prediksi, sisanya 0
    pred_array = np.zeros((len(preds_scaled), len(feature_cols)))
    pred_array[:, close_idx] = preds_scaled

    preds = scaler.inverse_transform(pred_array)[:, close_idx]

    return preds
