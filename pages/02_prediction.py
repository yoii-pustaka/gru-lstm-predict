import os
import streamlit as st
from scripts.data_loader import get_historical_data
from scripts.feature_engineering import create_features
from scripts.model_predict import predict_next_days
from scripts.plot import plot_prediction_vs_actual
from scripts.utils import evaluate_predictions

st.title("🔮 Prediksi Harga Saham")

available_tickers = ["BBCA.JK", "ANTM.JK", "TLKM.JK", "UNVR.JK"]
ticker = st.selectbox("Pilih Ticker Saham", options=available_tickers)
model_type = st.selectbox("Pilih Model", options=["LSTM", "GRU"])
days_ahead = st.number_input("Prediksi Berapa Hari ke Depan?", min_value=1, max_value=30, value=1)
seq_length = 60

if ticker:
    df = get_historical_data(ticker)
    if df is not None and not df.empty:
        st.subheader("📈 Data Historis Saham")
        st.dataframe(df.tail())

        df_feat = create_features(df)

        # ⛏️ Cek apakah model sudah tersedia
        model_path = f"models/{ticker}_{model_type}_model.h5"
        scaler_path = f"models/{ticker}_{model_type}_scaler.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.warning(f"Model '{model_type}' untuk {ticker} belum tersedia. Silakan latih terlebih dahulu di halaman 'Training'.")
        else:
            # 🔮 Prediksi hanya dilakukan jika model tersedia dan tombol ditekan
            if st.button("Prediksi"):
                try:
                    preds = predict_next_days(df_feat, ticker, model_type=model_type, days_ahead=days_ahead, seq_length=seq_length)
                    st.subheader(f"📊 Prediksi Harga Penutupan {days_ahead} Hari ke Depan")
                    for i, p in enumerate(preds, 1):
                        st.write(f"Hari {i}: {p:.2f}")

                    actual = df['Close'][-days_ahead:].values
                    if len(actual) < days_ahead:
                        st.warning("Data aktual kurang dari jumlah hari prediksi, evaluasi mungkin tidak akurat.")

                    if len(actual) > 0:
                        mae, rmse, mape = evaluate_predictions(actual, preds)
                        st.subheader("📉 Evaluasi Prediksi")
                        st.write(f"MAE: {mae:.2f}")
                        st.write(f"RMSE: {rmse:.2f}")
                        st.write(f"MAPE: {mape:.2f}%")

                        fig = plot_prediction_vs_actual(actual, preds, ticker)
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Prediksi gagal: {e}")
    else:
        st.error("Gagal memuat data saham.")
