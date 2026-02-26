import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- 1. Memuat model dan scaler yang telah disimpan ---
@st.cache_resource # Cache the model and scaler to avoid reloading on each rerun
def load_resources():
    try:
        with open('best_model_gradient_boosting.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('feature_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Pastikan file model 'best_model_gradient_boosting.pkl' dan scaler 'feature_scaler.pkl' ada di direktori yang sama.")
        st.stop()

loaded_model, loaded_scaler = load_resources()

# --- 2. Inisialisasi Label Encoders dengan kategori yang digunakan saat training ---
# Kategori untuk Pendidikan
le_pendidikan = LabelEncoder()
le_pendidikan.fit(['D3', 'S1', 'SMA', 'SMK']) # Pastikan urutan dan nilai sama dengan saat fitting

# Kategori untuk Jurusan
le_jurusan = LabelEncoder()
le_jurusan.fit(['administrasi', 'desain grafis', 'otomotif', 'teknik las', 'teknik listrik']) # Pastikan urutan dan nilai sama dengan saat fitting

# --- 3. Definisi urutan kolom fitur yang diharapkan oleh model ---
# Ini harus sama persis dengan `feature_cols` yang digunakan saat training
feature_cols = [
    'Pendidikan',
    'Jurusan',
    'Jenis_Kelamin_Laki-laki',
    'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja',
    'Status_Bekerja_Sudah Bekerja',
    'Usia',
    'Durasi_Jam',
    'Nilai_Ujian'
]

# --- 4. Streamlit UI ---
st.title("Prediksi Gaji Pertama Setelah Pelatihan Vokasi")
st.write("Aplikasi ini memprediksi estimasi gaji pertama (dalam juta Rupiah) berdasarkan profil peserta pelatihan vokasi.")

st.sidebar.header("Input Data Peserta")

usia = st.sidebar.number_input("Usia (Tahun)", min_value=18, max_value=60, value=25)
durasi_jam = st.sidebar.number_input("Durasi Pelatihan (Jam)", min_value=20, max_value=100, value=60)
nilai_ujian = st.sidebar.number_input("Nilai Ujian (Skala 0-100)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
pendidikan = st.sidebar.selectbox("Pendidikan Terakhir", le_pendidikan.classes_)
jurusan = st.sidebar.selectbox("Jurusan Pelatihan", le_jurusan.classes_)
jenis_kelamin = st.sidebar.radio("Jenis Kelamin", ['Laki-laki', 'Wanita'])
status_bekerja = st.sidebar.radio("Status Bekerja", ['Belum Bekerja', 'Sudah Bekerja'])

# Tombol Prediksi
if st.sidebar.button("Prediksi Gaji"):
    # --- 5. Pra-pemrosesan data input baru (sesuai dengan proses training) ---

    # Membuat DataFrame dari input pengguna
    input_data = pd.DataFrame([{
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }])

    # 5.1. Label Encoding untuk kolom 'Pendidikan' dan 'Jurusan'
    input_data['Pendidikan'] = le_pendidikan.transform(input_data['Pendidikan'])
    input_data['Jurusan'] = le_jurusan.transform(input_data['Jurusan'])

    # 5.2. One-Hot Encoding untuk kolom 'Jenis_Kelamin' dan 'Status_Bekerja'
    df_onehot_temp = pd.get_dummies(input_data[['Jenis_Kelamin', 'Status_Bekerja']])
    df_onehot_temp = df_onehot_temp.astype(int)

    # Define all possible one-hot encoded columns based on training data
    training_onehot_cols_expected = [
        'Jenis_Kelamin_Laki-laki',
        'Jenis_Kelamin_Wanita',
        'Status_Bekerja_Belum Bekerja',
        'Status_Bekerja_Sudah Bekerja'
    ]

    # Create a DataFrame with all expected one-hot columns and fill with 0, then update with actual values
    df_onehot_processed = pd.DataFrame(0, index=input_data.index, columns=training_onehot_cols_expected)
    for col in df_onehot_temp.columns:
        if col in df_onehot_processed.columns:
            df_onehot_processed[col] = df_onehot_temp[col]

    # 5.3. Menggabungkan semua fitur dalam urutan yang benar (sesuai dengan feature_cols)
    X_processed_dict = {
        'Pendidikan': input_data['Pendidikan'].iloc[0],
        'Jurusan': input_data['Jurusan'].iloc[0],
        'Jenis_Kelamin_Laki-laki': df_onehot_processed['Jenis_Kelamin_Laki-laki'].iloc[0],
        'Jenis_Kelamin_Wanita': df_onehot_processed['Jenis_Kelamin_Wanita'].iloc[0],
        'Status_Bekerja_Belum Bekerja': df_onehot_processed['Status_Bekerja_Belum Bekerja'].iloc[0],
        'Status_Bekerja_Sudah Bekerja': df_onehot_processed['Status_Bekerja_Sudah Bekerja'].iloc[0],
        'Usia': input_data['Usia'].iloc[0],
        'Durasi_Jam': input_data['Durasi_Jam'].iloc[0],
        'Nilai_Ujian': input_data['Nilai_Ujian'].iloc[0]
    }
    X_processed = pd.DataFrame([X_processed_dict], columns=feature_cols)

    # Ensure all columns are numeric (float) before scaling
    X_processed = X_processed.astype(float)

    # 5.4. Scaling fitur menggunakan loaded_scaler
    X_scaled = loaded_scaler.transform(X_processed)

    # Convert back to DataFrame for prediction (model expects DataFrame or array with column names if trained with DataFrame)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    # --- 6. Melakukan prediksi ---
    prediksi_gaji = loaded_model.predict(X_scaled_df)[0]

    # --- 7. Menampilkan hasil prediksi ---
    st.subheader("Hasil Prediksi")
    st.metric("Estimasi Gaji Pertama", f"{prediksi_gaji:.2f} Juta Rupiah")

    st.write("---")
    st.info("Catatan: Prediksi ini adalah estimasi berdasarkan model yang dilatih. Masukan akan diproses dengan Label Encoding untuk Pendidikan dan Jurusan, One-Hot Encoding untuk Jenis Kelamin dan Status Bekerja, lalu diskala menggunakan StandardScaler sebelum dimasukkan ke model prediksi.")
