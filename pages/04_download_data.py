import streamlit as st
import os
from scripts.data_loader import download_and_cache_data

available_tickers = ["BBCA.JK", "ANTM.JK", "TLKM.JK", "UNVR.JK"]
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

st.title("Download Data Saham")

ticker = st.selectbox("Pilih Ticker Saham untuk Download Data", options=available_tickers)

if st.button("Download Data CSV"):
    with st.spinner(f"Downloading data for {ticker}..."):
        df = download_and_cache_data(ticker)
        if df is not None:
            st.success(f"Data untuk {ticker} berhasil didownload dan disimpan.")
        else:
            st.error("Gagal mendownload data.")
