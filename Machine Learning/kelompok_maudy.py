import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load('ML_kemiskinan.pkl')

# Daftar provinsi yang valid
provinsi_list = ['Aceh', 'Sumatera Utara', 'Sumatera Barat	', 'Riau', 'Jambi', 'Sumatera Selatan']

# Membuat encoder baru
encoder = LabelEncoder()
encoder.fit(provinsi_list)  # Melatih encoder dengan semua provinsi yang valid

st.title("Prediksi Tingkat Kemiskinan Tertinggi")

# Input dari pengguna
Provinsi = st.selectbox("Provinsi", options=provinsi_list)
jumlah_penduduk = st.number_input("Jumlah Penduduk", min_value=0, value=50)

if st.button("Prediksi"):
    if Provinsi and jumlah_penduduk > 0:  # Pastikan input tidak kosong
        # Encode provinsi
        try:
            provinsi_encoded = encoder.transform([Provinsi])  # Encode provinsi
            # Data input berbentuk array 2D
            data = np.array([[provinsi_encoded[0], jumlah_penduduk]])
            pred_label = model.predict(data)[0]
            pred_kemiskinan = encoder.inverse_transform([pred_label])[0]
            st.success(f"Prediksi kategori kemiskinan untuk provinsi {Provinsi} adalah: {Provinsi}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.error("Silakan masukkan provinsi dan jumlah penduduk yang valid.")