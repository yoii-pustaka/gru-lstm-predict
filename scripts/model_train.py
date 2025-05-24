import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from scripts.utils import save_model, save_scaler

def prepare_data(df, feature_cols, seq_length):
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    x, y = [], []
    for i in range(seq_length, len(data_scaled)):
        x.append(data_scaled[i-seq_length:i])
        y.append(data_scaled[i, feature_cols.index('Close')])
    return np.array(x), np.array(y), scaler

def build_model(seq_length, feature_count, model_type='LSTM', units=50):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=units, input_shape=(seq_length, feature_count)))
    elif model_type == 'GRU':
        model.add(GRU(units=units, input_shape=(seq_length, feature_count)))
    else:
        raise ValueError("model_type must be 'LSTM' or 'GRU'")
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(df, ticker, model_type='LSTM', epochs=50, seq_length=60):
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    x, y, scaler = prepare_data(df, feature_cols, seq_length)
    model = build_model(seq_length, len(feature_cols), model_type)
    model.fit(x, y, epochs=epochs, batch_size=32)
    save_model(model, ticker, model_type)
    save_scaler(scaler, ticker, model_type)
    return model, scaler
