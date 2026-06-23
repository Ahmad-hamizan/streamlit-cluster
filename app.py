import streamlit as st
import pandas as pd
import joblib
import os

# 1. SETTING HALAMAN STREAMLIT
st.set_page_config(page_title="Dashboard Analisis Bank Sampah", layout="wide")

st.title("🎯 Dashboard Evaluasi Model & Sebaran Cluster Bank Sampah")
st.write("Aplikasi web ini membaca model K-Means yang telah dilatih (.pkl) beserta visualisasinya.")

# 2. LOAD DATA DARI JUPYTER NOTEBOOK
@st.cache_data
def load_base_data():
    df_mentah = pd.read_csv('bank_sampah.csv')
    X_numeric = joblib.load('data_cluster_wilayah.pkl')
    
    # Map cluster dari X_numeric ke df_mentah menggunakan nama kelurahan
    df_mentah['Cluster'] = df_mentah['bps_desa_kelurahan'].map(X_numeric['Cluster'])
    
    # Koordinat GPS Asli Wilayah Kelurahan di Cibeunying Kidul, Bandung
    koordinat_desa = {
        'CICADAS': [-6.9061, 107.6436],
        'CIKUTRA': [-6.8974, 107.6369],
        'PADASUKA': [-6.8994, 107.6472],
        'PASIRLAYUNG': [-6.8911, 107.6534],
        'SUKAMAJU': [-6.8952, 107.6415],
        'SUKAPADA': [-6.8931, 107.6461]
    }
    
    df_mentah['latitude'] = df_mentah['bps_desa_kelurahan'].map(lambda x: koordinat_desa.get(x, [-6.9175, 107.6191])[0])
    df_mentah['longitude'] = df_mentah['bps_desa_kelurahan'].map(lambda x: koordinat_desa.get(x, [-6.9175, 107.6191])[1])
    
    # Set warna hex unik untuk st.map
    warna_cluster = {0: '#FF4B4B', 1: '#00E676', 2: '#29B6F6'}
    df_mentah['warna'] = df_mentah['Cluster'].map(warna_cluster)
    
    return df_mentah, X_numeric

try:
    df, X_numeric = load_base_data()

    # TAB MENU UTAMA
    tab1, tab2 = st.tabs(["🗺️ Peta & Filter Data", "📊 Evaluasi Model (Elbow & 2D)"])

    # --- TAB 1: PETA DAN FILTER DATA ---
    with tab1:
        st.subheader("Sebaran Geografis Bank Sampah")
        col_input, col_peta = st.columns([1, 2])
        
        with col_input:
            pilihan_desa = st.selectbox("Pilih Kelurahan / Desa:", ["Semua Desa"] + list(df['bps_desa_kelurahan'].unique()))
            
            if pilihan_desa != "Semua Desa":
                unit_tersedia = df[df['bps_desa_kelurahan'] == pilihan_desa]['nama_unit_bank_sampah'].unique()
            else:
                unit_tersedia = df['nama_unit_bank_sampah'].unique()
                
            pilihan_unit = st.selectbox("Pilih Nama Unit Bank Sampah:", ["Semua Unit"] + list(unit_tersedia))
            
            st.info("💡 **Legenda Warna Titik Peta:**\n"
                    "- 🔴 **Cluster 0**\n"
                    "- 🟢 **Cluster 1**\n"
                    "- 🔵 **Cluster 2**")
            
        # Terapkan filter data
        df_filtered = df.copy()
        if pilihan_desa != "Semua Desa":
            df_filtered = df_filtered[df_filtered['bps_desa_kelurahan'] == pilihan_desa]
        if pilihan_unit != "Semua Unit":
            df_filtered = df_filtered[df_filtered['nama_unit_bank_sampah'] == pilihan_unit]

        with col_peta:
            # Menampilkan peta interaktif asli wilayah bandung
            st.map(df_filtered, latitude='latitude', longitude='longitude', color='warna', size=50)

        st.write("### 📋 Tabel Hasil Pemetaan")
        st.dataframe(df_filtered[['bps_desa_kelurahan', 'nama_unit_bank_sampah', 'Cluster']].reset_index(drop=True), use_container_width=True)

    # --- TAB 2: EVALUASI MODEL DARI JUPYTER ---
    with tab2:
        st.subheader("Hasil Evaluasi dan Visualisasi dari Jupyter Notebook")
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**1. Analisis Penentuan Jumlah Cluster Optimal**")
            if os.path.exists('elbow_plot.png'):
                st.image('elbow_plot.png', caption='Grafik Elbow Method (Inertia berkurang drastis di k=3)')
            else:
                st.warning("File elbow_plot.png tidak ditemukan. Jalankan kode di Jupyter dulu!")
                
        with c2:
            st.write("**2. Visualisasi Pembagian Cluster Data Penduduk**")
            if os.path.exists('kmeans_2d_plot.png'):
                st.image('kmeans_2d_plot.png', caption='Peta Sebaran Cluster Berdasarkan Total Unit & Durasi')
            else:
                st.warning("File kmeans_2d_plot.png tidak ditemukan. Jalankan kode di Jupyter dulu!")

except FileNotFoundError:
    st.error("Pastikan file `bank_sampah.csv` dan file `.pkl` berada dalam satu folder yang sama dengan file `app.py` ini.")