import streamlit as st
from scripts.data_loader import get_historical_data
from scripts.feature_engineering import create_features
from scripts.model_train import train_model

st.title("🧠 Training Model")

available_tickers = ["BBCA.JK", "ANTM.JK", "TLKM.JK", "UNVR.JK"]
ticker = st.selectbox("Pilih Ticker Saham", options=available_tickers)
model_type = st.selectbox("Pilih Model", options=["LSTM", "GRU"])
seq_length = 60

if ticker:
    df = get_historical_data(ticker)
    if df is not None and not df.empty:
        st.subheader("Data Historis")
        st.dataframe(df.tail())

        df_feat = create_features(df)
        st.subheader("Data dengan Fitur")
        st.dataframe(df_feat.tail())

        if st.button("Train Model"):
            with st.spinner("Melatih model, harap tunggu..."):
                train_model(df_feat, ticker, model_type=model_type, epochs=10, seq_length=seq_length)
            st.success(f"Model untuk {ticker} berhasil dilatih dan disimpan!")
    else:
        st.error("Gagal memuat data saham.")
