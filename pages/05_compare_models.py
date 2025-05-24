import streamlit as st
from scripts.data_loader import get_historical_data
from scripts.feature_engineering import create_features
from scripts.model_predict import predict_next_days
from scripts.plot import plot_compare_lstm_gru
from scripts.utils import evaluate_predictions

st.title("Perbandingan Prediksi LSTM dan GRU")

available_tickers = ["BBCA.JK", "ANTM.JK", "TLKM.JK", "UNVR.JK"]
ticker = st.selectbox("Pilih Ticker Saham", options=available_tickers)
days_ahead = st.number_input("Jumlah Hari Prediksi", min_value=1, max_value=30, value=1)
seq_length = 60

if ticker:
    df = get_historical_data(ticker)
    if df is not None and not df.empty:
        df_feat = create_features(df)
        if st.button(f"Bandingkan Prediksi LSTM dan GRU {days_ahead} Hari ke Depan"):
            try:
                preds_lstm = predict_next_days(df_feat, ticker, model_type='LSTM', days_ahead=days_ahead, seq_length=seq_length)
                preds_gru = predict_next_days(df_feat, ticker, model_type='GRU', days_ahead=days_ahead, seq_length=seq_length)

                actual = df['Close'][-days_ahead:].values
                if len(actual) < days_ahead:
                    st.warning("Data aktual kurang dari jumlah hari prediksi, grafik dan evaluasi mungkin kurang akurat.")

                fig = plot_compare_lstm_gru(actual, preds_lstm, preds_gru, ticker)
                st.pyplot(fig)

                # Evaluasi
                mae_lstm, rmse_lstm, mape_lstm = evaluate_predictions(actual, preds_lstm)
                mae_gru, rmse_gru, mape_gru = evaluate_predictions(actual, preds_gru)

                st.subheader("Evaluasi Model")
                st.write(f"LSTM - MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}, MAPE: {mape_lstm:.2f}%")
                st.write(f"GRU  - MAE: {mae_gru:.2f}, RMSE: {rmse_gru:.2f}, MAPE: {mape_gru:.2f}%")

            except Exception as e:
                st.error(f"Gagal prediksi atau plot: {e}")
    else:
        st.error("Gagal mengambil data saham.")
