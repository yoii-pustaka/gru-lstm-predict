import streamlit as st

st.title("Tentang Aplikasi")

st.write("""
Aplikasi ini dibuat untuk memprediksi harga saham menggunakan model LSTM dan GRU.
\n
Fitur:
- Download data saham dari Yahoo Finance (dengan caching lokal)
- Training model
- Prediksi harga saham dan perbandingan dengan harga aktual
\n
Dibuat oleh: [Nama Kamu]
""")
