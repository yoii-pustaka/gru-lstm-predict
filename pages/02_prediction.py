import streamlit as st
import os
import numpy as np
from scripts.data_loader import get_historical_data
from scripts.feature_engineering import create_features
from scripts.model_predict import predict_next_days
from scripts.plot import plot_prediction_vs_actual
from scripts.utils import load_model_file, load_scaler_file, evaluate_predictions
from scripts.model_train import train_test_split_close_only


st.title("🔮 Prediksi & Evaluasi Harga Saham (Close Only)")

available_tickers = ["BBCA.JK", "ANTM.JK", "TLKM.JK", "UNVR.JK"]
ticker = st.selectbox("📌 Pilih Ticker Saham", options=available_tickers)
model_type = st.selectbox("📂 Pilih Model", options=["LSTM", "GRU"])
seq_length = 60

df = get_historical_data(ticker)

if df is not None and not df.empty:
    df_feat = create_features(df)
    st.subheader("📊 Data Historis")
    st.dataframe(df_feat.tail())

    model_path = f"models/{ticker}_{model_type}_model.h5"
    scaler_path = f"models/{ticker}_{model_type}_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning(f"Model '{model_type}' untuk {ticker} belum tersedia. Silakan latih dahulu.")
    else:
        st.header("✅ Evaluasi Model (80% Train / 20% Test Split)")

        x_train, y_train, x_test, y_test, scaler = train_test_split_close_only(df_feat, ['Close'], seq_length)

        model = load_model_file(ticker, model_type)

        y_pred_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)

        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
        y_pred_inv = scaler.inverse_transform(y_pred_scaled)[:, 0]

        mae, rmse, mape = evaluate_predictions(y_test_inv, y_pred_inv)

        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")

        fig = plot_prediction_vs_actual(y_test_inv, y_pred_inv, ticker)
        st.pyplot(fig)


        st.header("🔮 Prediksi Masa Depan (Forward)")
        days_option = st.selectbox("Pilih Horizon Prediksi", options=[1, 7, 15, 30], index=0)

        if st.button("Prediksi ke Depan"):
            preds = predict_next_days(df_feat, ticker, model_type, days_ahead=days_option, seq_length=seq_length)

            st.subheader(f"📈 Prediksi Harga Close {days_option} Hari ke Depan")
            for i, p in enumerate(preds, 1):
                st.write(f"Hari {i}: {p:.2f}")
