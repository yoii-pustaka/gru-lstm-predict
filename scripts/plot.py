import matplotlib.pyplot as plt

def plot_stock_chart(df, selected_cols, ticker):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for col in selected_cols:
        ax.plot(df.index, df[col], label=col)

    ax.set_title(f"Harga Saham {ticker}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Nilai")
    ax.legend()
    ax.grid(True)

    return fig


def plot_prediction_vs_actual(actual, predicted, ticker):
    """
    Plot perbandingan harga aktual dan prediksi.
    actual dan predicted harus array/list dengan panjang sama.
    """
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(actual, label='Harga Aktual', marker='o')
    ax.plot(predicted, label='Harga Prediksi', marker='x', linestyle='--')
    ax.set_title(f"Perbandingan Harga Aktual vs Prediksi untuk {ticker}")
    ax.set_xlabel("Hari ke depan")
    ax.set_ylabel("Harga Close")
    ax.legend()
    ax.grid(True)
    return fig

def plot_compare_lstm_gru(actual, preds_lstm, preds_gru, ticker):
    days = range(1, len(actual) + 1)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(days, actual, label="Harga Aktual", marker='o')
    ax.plot(days, preds_lstm, label="Prediksi LSTM", marker='x')
    ax.plot(days, preds_gru, label="Prediksi GRU", marker='^')

    ax.set_title(f"Perbandingan Prediksi LSTM vs GRU - {ticker}")
    ax.set_xlabel("Hari ke-")
    ax.set_ylabel("Harga Penutupan")
    ax.legend()
    ax.grid(True)
    return fig

def plot_compare_lstm_gru(actual, preds_lstm, preds_gru, ticker):
    plt.figure(figsize=(10,6))
    days = range(1, len(actual)+1)

    plt.plot(days, actual, marker='o', linestyle='-', color='black', label='Harga Aktual')
    plt.plot(days, preds_lstm, marker='x', linestyle='--', color='blue', label='Prediksi LSTM')
    plt.plot(days, preds_gru, marker='s', linestyle='--', color='red', label='Prediksi GRU')

    plt.title(f"Perbandingan Prediksi Harga Saham {ticker}")
    plt.xlabel("Hari ke Depan")
    plt.ylabel("Harga Penutupan")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt