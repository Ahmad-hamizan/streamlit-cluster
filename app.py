import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# 1. SETTING HALAMAN STREAMLIT
st.set_page_config(page_title="Dashboard Cluster Bank Sampah", layout="wide")

st.title("🎯 Dashboard Pencarian Kelompok Cluster Bank Sampah")
st.write("Aplikasi ini otomatis melakukan clustering pada tingkat Kelurahan/Desa, "
         "lalu memetakan hasilnya kembali ke setiap unit Bank Sampah.")

# 2. FUNGSI UNTUK MEMBACA & MEMPROSES DATA SEARA OTOMATIS
@st.cache_data
def load_and_cluster_data():
    # Membaca dataset asli yang kamu upload
    df = pd.read_csv('bank_sampah.csv')
    
    # Membuat dataframe hasil agregasi (X_numeric) seperti di Notebook-mu
    X_numeric = df.groupby('bps_desa_kelurahan').agg(
        total_unit=('id', 'count'),
        min_year=('tahun', 'min'),
        max_year=('tahun', 'max')
    )
    # Menghitung durasi operasional
    X_numeric['durasi_tahun'] = X_numeric['max_year'] - X_numeric['min_year'] + 1
    
    # Menjalankan K-Means Clustering (sesuai jumlah cluster = 3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    X_numeric['Cluster'] = kmeans.fit_predict(X_numeric[['total_unit', 'durasi_tahun']])
    
    # MEMAKAI TRICK MAP KAMU: Gabungkan kembali ke dataframe asli (df)
    df['Cluster'] = df['bps_desa_kelurahan'].map(X_numeric['Cluster'])
    
    return df, X_numeric

# Menjalankan fungsi pemrosesan data
try:
    df, X_numeric = load_and_cluster_data()

    # 3. FITUR INTERAKTIF: PENCARIAN & FILTER
    st.subheader("🔍 Cari Data & Filter Wilayah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pilihan_desa = st.selectbox(
            "Pilih Kelurahan / Desa:",
            ["Semua Desa"] + list(df['bps_desa_kelurahan'].unique())
        )
        
    with col2:
        # Menyesuaikan pilihan nama unit berdasarkan desa yang dipilih
        if pilihan_desa != "Semua Desa":
            unit_tersedia = df[df['bps_desa_kelurahan'] == pilihan_desa]['nama_unit_bank_sampah'].unique()
        else:
            unit_tersedia = df['nama_unit_bank_sampah'].unique()
            
        pilihan_unit = st.selectbox(
            "Pilih Nama Unit Bank Sampah:",
            ["Semua Unit"] + list(unit_tersedia)
        )

    # Memfilter dataframe berdasarkan input dari user
    df_hasil = df.copy()
    if pilihan_desa != "Semua Desa":
        df_hasil = df_hasil[df_hasil['bps_desa_kelurahan'] == pilihan_desa]
    if pilihan_unit != "Semua Unit":
        df_hasil = df_hasil[df_hasil['nama_unit_bank_sampah'] == pilihan_unit]

    # 4. TAMPILKAN TABEL HASIL MAPPING (Sesuai Request Kamu)
    st.subheader("📋 Hasil Pemetaan Cluster Bank Sampah")
    
    # Mengambil kolom sesuai contoh yang kamu inginkan
    tabel_tampil = df_hasil[['bps_desa_kelurahan', 'nama_unit_bank_sampah', 'Cluster']].reset_index(drop=True)
    
    st.dataframe(tabel_tampil, use_container_width=True)

    # 5. INFORMASI TAMBAHAN: KARAKTERISTIK TIAP CLUSTER DI WILAYAH
    st.write("---")
    st.subheader("📊 Tabel Referensi Nilai Agregasi Wilayah (X_numeric)")
    st.dataframe(X_numeric[['total_unit', 'durasi_tahun', 'Cluster']], use_container_width=True)

except FileNotFoundError:
    st.error("File `bank_sampah.csv` tidak ditemukan! Pastikan file CSV tersebut diletakkan di dalam folder yang sama dengan file `app.py` ini.")