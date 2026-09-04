import streamlit as st
import pandas as pd
import re
import string
import pickle
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Analisis Sentimen Blibli",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# JUDUL APLIKASI
# ============================================================

st.title("📊 Analisis Sentimen Pengguna Aplikasi Blibli")

st.markdown(
    """
    **Analisis Sentimen Menggunakan Text Mining dan Naïve Bayes**

    Aplikasi ini digunakan untuk melakukan preprocessing,
    klasifikasi sentimen, visualisasi hasil, dan pengunduhan
    hasil analisis ulasan pengguna aplikasi Blibli.
    """
)

st.divider()

# ============================================================
# FUNGSI PREPROCESSING
# ============================================================

def cleaning(text):
    text = str(text)

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def case_folding(text):
    return str(text).lower()


def tokenizing(text):
    return str(text).split()


def stopword_removal(tokens):
    stopwords = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu",
        "untuk", "dengan", "pada", "adalah", "saya", "aku",
        "nya", "ga", "gak", "nggak", "tidak", "ada", "jadi",
        "karena", "atau", "juga", "sudah", "sangat", "lebih",
        "bisa", "dalam", "akan", "se", "aja"
    }

    return [
        word for word in tokens
        if word not in stopwords
    ]


def stemming(tokens):
    return tokens


def preprocess_text(text):
    text = cleaning(text)
    text = case_folding(text)
    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    tokens = stemming(tokens)

    return " ".join(tokens)


# ============================================================
# SESSION STATE
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = None

if "hasil" not in st.session_state:
    st.session_state.hasil = None

if "label_manual" not in st.session_state:
    st.session_state.label_manual = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📌 Menu")

menu = st.sidebar.radio(
    "Pilih Menu:",
    [
        "Beranda",
        "Upload Dataset",
        "Preprocessing",
        "Analisis Sentimen",
        "Visualisasi"
    ]
)


# ============================================================
# BERANDA
# ============================================================

if menu == "Beranda":

    st.subheader("Tentang Aplikasi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Metode", "Naïve Bayes")

    with col2:
        st.metric("Jenis Analisis", "Sentimen")

    with col3:
        st.metric("Sumber Data", "Google Play Store")

    st.info(
        """
        Aplikasi ini merupakan interface untuk membantu proses
        analisis sentimen ulasan pengguna aplikasi Blibli.
        """
    )

    st.markdown("### Alur Penggunaan")

    st.write("1. Upload dataset ulasan.")
    st.write("2. Periksa data.")
    st.write("3. Jalankan preprocessing.")
    st.write("4. Upload data labeling manual dan model.")
    st.write("5. Jalankan klasifikasi sentimen.")
    st.write("6. Lihat hasil dan visualisasi.")
    st.write("7. Download hasil analisis.")


# ============================================================
# UPLOAD DATASET
# ============================================================

elif menu == "Upload Dataset":

    st.subheader("📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.data = df
            st.session_state.hasil = None

            st.success("✅ Dataset berhasil diunggah.")

            st.write("### Informasi Dataset")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Jumlah Baris", len(df))

            with col2:
                st.metric("Jumlah Kolom", len(df.columns))

            st.write("### Preview Dataset")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.write("### Nama Kolom")

            st.write(list(df.columns))

        except Exception as e:

            st.error(f"Dataset gagal dibaca: {e}")


# ============================================================
# PREPROCESSING
# ============================================================

elif menu == "Preprocessing":

    st.subheader("🧹 Preprocessing Data")

    if st.session_state.data is None:

        st.warning("Silakan upload dataset terlebih dahulu.")

    else:

        df = st.session_state.data.copy()

        st.write("### Pilih Kolom Ulasan")

        text_column = st.selectbox(
            "Kolom teks ulasan:",
            df.columns
        )

        if st.button("▶ Jalankan Preprocessing"):

            with st.spinner("Sedang melakukan preprocessing..."):

                df["teks_asli"] = (
                    df[text_column]
                    .astype(str)
                )

                df["cleaning"] = (
                    df["teks_asli"]
                    .apply(cleaning)
                )

                df["case_folding"] = (
                    df["cleaning"]
                    .apply(case_folding)
                )

                df["tokenizing"] = (
                    df["case_folding"]
                    .apply(tokenizing)
                )

                df["stopword_removal"] = (
                    df["tokenizing"]
                    .apply(stopword_removal)
                )

                df["stemming"] = (
                    df["stopword_removal"]
                    .apply(stemming)
                )

                df["teks_bersih"] = (
                    df["stemming"]
                    .apply(lambda x: " ".join(x))
                )

                st.session_state.data = df

            st.success("✅ Preprocessing berhasil.")

            st.write("### Hasil Preprocessing")

            st.dataframe(
                df[
                    [
                        text_column,
                        "cleaning",
                        "case_folding",
                        "tokenizing",
                        "stopword_removal",
                        "stemming",
                        "teks_bersih"
                    ]
                ].head(20),
                use_container_width=True
            )

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Hasil Preprocessing",
                data=csv,
                file_name="hasil_preprocessing_blibli.csv",
                mime="text/csv"
            )


# ============================================================
# ANALISIS SENTIMEN
# ============================================================

elif menu == "Analisis Sentimen":

    st.title("🤖 Analisis Sentimen Naïve Bayes")

    if st.session_state.data is None:

        st.warning(
            "Silakan upload dataset terlebih dahulu."
        )
        st.stop()

    df = st.session_state.data.copy()

    # --------------------------------------------------------
    # CEK KOLOM ULASAN
    # --------------------------------------------------------

    if "h3YV2d" not in df.columns:

        st.error(
            "Kolom ulasan 'h3YV2d' tidak ditemukan pada dataset."
        )
        st.stop()

    # --------------------------------------------------------
    # UPLOAD LABEL MANUAL
    # --------------------------------------------------------

    st.subheader("1️⃣ Upload Data Labeling Manual")

    st.write(
        "Upload file Excel yang berisi 600 data yang telah "
        "diberi label manual."
    )

    uploaded_label = st.file_uploader(
        "Upload file labeling manual (.xlsx)",
        type=["xlsx"],
        key="label_manual"
    )

    # --------------------------------------------------------
    # UPLOAD MODEL
    # --------------------------------------------------------

    st.subheader("2️⃣ Upload Model Naïve Bayes")

    uploaded_model = st.file_uploader(
        "Upload model Naïve Bayes (.pkl)",
        type=["pkl"],
        key="model_nb"
    )

    # --------------------------------------------------------
    # PROSES
    # --------------------------------------------------------

    if uploaded_label is not None and uploaded_model is not None:

        try:

            # ================================================
            # BACA DATA LABELING
            # ================================================

            label_df = pd.read_excel(
                uploaded_label
            )

            # Cari kolom ulasan
            kolom_ulasan_label = None

            for kolom in label_df.columns:

                if str(kolom).strip().lower() in [
                    "ulasan",
                    "review",
                    "reviews",
                    "teks"
                ]:
                    kolom_ulasan_label = kolom
                    break

            if kolom_ulasan_label is None:

                st.error(
                    "Kolom ulasan pada file labeling tidak ditemukan."
                )
                st.stop()

            # Cari kolom sentimen
            kolom_sentimen = None

            for kolom in label_df.columns:

                if str(kolom).strip().lower() in [
                    "sentimen",
                    "sentiment",
                    "label"
                ]:
                    kolom_sentimen = kolom
                    break

            if kolom_sentimen is None:

                st.error(
                    "Kolom sentimen pada file labeling tidak ditemukan."
                )
                st.stop()

            # Pastikan jumlah data manual
            if len(label_df) != 600:

                st.warning(
                    f"File labeling berisi {len(label_df)} data. "
                    "Penelitian menggunakan 600 data manual."
                )

            # ================================================
            # NORMALISASI TEKS UNTUK MATCHING
            # ================================================

            def normalisasi_matching(text):

                text = str(text).lower().strip()

                text = re.sub(
                    r"\s+",
                    " ",
                    text
                )

                return text

            dataset_match = (
                df["h3YV2d"]
                .fillna("")
                .astype(str)
                .apply(normalisasi_matching)
            )

            label_match = (
                label_df[kolom_ulasan_label]
                .fillna("")
                .astype(str)
                .apply(normalisasi_matching)
            )

            # ================================================
            # MATCHING 600 DATA MANUAL
            # ================================================

            label_mapping = {}

            for i, teks in enumerate(label_match):

                if teks not in label_mapping:
                    label_mapping[teks] = []

                label_mapping[teks].append(
                    label_df.iloc[i][kolom_sentimen]
                )

            label_counter = {}

            for teks, labels in label_mapping.items():

                label_counter[teks] = 0

            label_manual_hasil = [None] * len(df)

            for i, teks in enumerate(dataset_match):

                if teks in label_mapping:

                    posisi = label_counter[teks]

                    if posisi < len(label_mapping[teks]):

                        label_manual_hasil[i] = (
                            label_mapping[teks][posisi]
                        )

                        label_counter[teks] += 1

            jumlah_manual = sum(
                x is not None
                for x in label_manual_hasil
            )

            st.info(
                f"Data manual yang berhasil dicocokkan: "
                f"**{jumlah_manual} data**"
            )

            if jumlah_manual != 600:

                st.warning(
                    "Jumlah data manual yang cocok belum 600. "
                    "Periksa kembali file dataset dan file labeling."
                )

            # ================================================
            # LOAD MODEL
            # ================================================

            model = pickle.load(
                uploaded_model
            )

            # ================================================
            # BUAT KOLOM SENTIMEN
            # ================================================

            df["sentimen"] = label_manual_hasil

            # ================================================
            # AMBIL DATA YANG BELUM DILABELI
            # ================================================

            indeks_belum_label = df[
                "sentimen"
            ].isna()

            jumlah_belum_label = indeks_belum_label.sum()

            st.info(
                f"Data yang akan diprediksi oleh Naïve Bayes: "
                f"**{jumlah_belum_label} data**"
            )

            # ================================================
            # PREDIKSI 460 DATA
            # ================================================

            if jumlah_belum_label > 0:

                teks_prediksi = (
                    df.loc[
                        indeks_belum_label,
                        "h3YV2d"
                    ]
                    .fillna("")
                    .astype(str)
                )

                # Jika preprocessing sudah dijalankan,
                # gunakan teks bersih.
                if "teks_bersih" in df.columns:

                    teks_model = (
                        df.loc[
                            indeks_belum_label,
                            "teks_bersih"
                        ]
                        .fillna("")
                        .astype(str)
                    )

                else:

                    teks_model = (
                        teks_prediksi
                        .apply(preprocess_text)
                    )

                prediksi = model.predict(
                    teks_model
                )

                prediksi = (
                    pd.Series(prediksi)
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.capitalize()
                    .values
                )

                df.loc[
                    indeks_belum_label,
                    "sentimen"
                ] = prediksi

            # ================================================
            # SIMPAN HASIL
            # ================================================

            df["sentimen"] = (
                df["sentimen"]
                .astype(str)
                .str.strip()
                .str.capitalize()
            )

            st.session_state.hasil = df

            st.success(
                "✅ Analisis sentimen berhasil dilakukan!"
            )

            # ================================================
            # RINGKASAN
            # ================================================

            st.subheader("📊 Ringkasan Sentimen")

            positif = (
                df["sentimen"] == "Positif"
            ).sum()

            netral = (
                df["sentimen"] == "Netral"
            ).sum()

            negatif = (
                df["sentimen"] == "Negatif"
            ).sum()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Positif",
                    positif
                )

            with col2:
                st.metric(
                    "Netral",
                    netral
                )

            with col3:
                st.metric(
                    "Negatif",
                    negatif
                )

            # ================================================
            # HASIL PREDIKSI
            # ================================================

            st.subheader("📋 Hasil Analisis")

            st.dataframe(
                df[
                    ["h3YV2d", "sentimen"]
                ],
                use_container_width=True
            )

            # ================================================
            # DOWNLOAD
            # ================================================

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Hasil Analisis",
                data=csv,
                file_name="HASIL_ANALISIS_SENTIMEN_BLIBLI.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Analisis gagal dilakukan: {e}"
            )


# ============================================================
# VISUALISASI
# ============================================================

elif menu == "Visualisasi":

    st.subheader("📊 Visualisasi Hasil Sentimen")

    if st.session_state.hasil is None:

        st.warning(
            "Jalankan analisis sentimen terlebih dahulu."
        )

    else:

        df = st.session_state.hasil

        if "sentimen" not in df.columns:

            st.warning(
                "Kolom sentimen belum tersedia."
            )

        else:

            jumlah = (
                df["sentimen"]
                .value_counts()
            )

            # Pastikan urutan kategori
            urutan = [
                "Positif",
                "Netral",
                "Negatif"
            ]

            jumlah = (
                jumlah
                .reindex(urutan, fill_value=0)
            )

            st.write(
                "### Grafik Distribusi Sentimen"
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            jumlah.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title(
                "Distribusi Sentimen Ulasan Pengguna Blibli"
            )

            ax.set_xlabel(
                "Sentimen"
            )

            ax.set_ylabel(
                "Jumlah Ulasan"
            )

            plt.xticks(
                rotation=0
            )

            st.pyplot(fig)

            st.write(
                "### Distribusi Sentimen"
            )

            tabel = pd.DataFrame(
                {
                    "Sentimen": jumlah.index,
                    "Jumlah": jumlah.values
                }
            )

            st.dataframe(
                tabel,
                use_container_width=True
            )

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Hasil Analisis",
                data=csv,
                file_name="HASIL_ANALISIS_SENTIMEN_BLIBLI.csv",
                mime="text/csv"
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Analisis Sentimen Pengguna Aplikasi Blibli"
)

st.sidebar.caption(
    "Text Mining & Naïve Bayes"
)
