import os
import pandas as pd
from datetime import datetime
from yahooquery import Ticker

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
DUMMY_DATA_PATH = os.path.join(DATA_DIR, 'dummy_data.csv')

def download_and_cache_data(ticker):
    try:
        ticker_obj = Ticker(ticker)
        today = datetime.today().strftime('%Y-%m-%d')
        hist = ticker_obj.history(start='2020-01-01', end=today)
        if hist.empty:
            print(f"Data kosong dari Yahooquery untuk {ticker}")
            return None
        
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index(level=0, drop=True)

        filename = f"{ticker}_data.csv"
        file_path = os.path.join(DATA_DIR, filename)
        hist.to_csv(file_path)
        print(f"Data untuk {ticker} berhasil didownload dan disimpan.")
        return hist
    except Exception as e:
        print(f"Gagal download data dari yahooquery: {e}")
        return None

def get_historical_data(ticker):
    filename = f"{ticker}_data.csv"
    file_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df.index.name = 'Date'
            df.columns = [col.capitalize() for col in df.columns]
            expected_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            for col in expected_cols:
                if col not in df.columns:
                    raise ValueError(f"Kolom '{col}' tidak ditemukan di file {filename}")
            return df
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    else:
        print(f"File {filename} tidak ditemukan di {DATA_DIR}")

    # fallback dummy data jika file tidak ada atau error
    if os.path.exists(DUMMY_DATA_PATH):
        try:
            df = pd.read_csv(DUMMY_DATA_PATH, index_col=0, parse_dates=True)
            return df
        except Exception as e2:
            print(f"Failed to load dummy data: {e2}")

    return pd.DataFrame()
