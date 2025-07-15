import numpy as np
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from scripts.utils import save_model, save_scaler


def prepare_data_close_only(df, seq_length):
    data = df[['Close']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    x, y = [], []
    for i in range(seq_length, len(data_scaled)):
        x.append(data_scaled[i - seq_length:i])
        y.append(data_scaled[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y, scaler


def build_model(seq_length, model_type='LSTM', units=50):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=units, input_shape=(seq_length, 1)))
    elif model_type == 'GRU':
        model.add(GRU(units=units, input_shape=(seq_length, 1)))
    else:
        raise ValueError("model_type must be 'LSTM' or 'GRU'")

    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_test_split_close_only(df, feature_cols, seq_length, split_ratio=0.8):
    data = df[feature_cols].values
    total_len = len(data)
    train_len = int(total_len * split_ratio)
    train_data = data[:train_len]
    test_data = data[train_len - seq_length:]  # overlap agar sequence terakhir tetap valid

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    def create_sequence(data_scaled):
        x, y = [], []
        for i in range(seq_length, len(data_scaled)):
            x.append(data_scaled[i - seq_length:i])
            y.append(data_scaled[i, 0])
        return np.array(x), np.array(y)

    x_train, y_train = create_sequence(train_scaled)
    x_test, y_test = create_sequence(test_scaled)

    return x_train, y_train, x_test, y_test, scaler


def train_model(df, ticker, model_type='LSTM', epochs=50, seq_length=60):
    feature_cols = ['Close']
    x_train, y_train, x_test, y_test, scaler = train_test_split_close_only(df, feature_cols, seq_length)

    model = build_model(seq_length, model_type)
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    save_model(model, ticker, model_type)
    save_scaler(scaler, ticker, model_type)

    # Simpan metadata fitur
    model_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(model_dir, exist_ok=True)
    feature_path = os.path.join(model_dir, f'{ticker}_{model_type}_features.json')
    with open(feature_path, 'w') as f:
        json.dump(['Close'], f)

    return model, scaler, x_test, y_test
