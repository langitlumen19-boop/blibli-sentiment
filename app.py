
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

    # Menghapus URL
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Menghapus mention
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Menghapus hashtag
    text = re.sub(
        r"#\w+",
        "",
        text
    )

    # Menghapus angka
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Menghapus tanda baca
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Menghapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def case_folding(text):

    return str(text).lower()


def tokenizing(text):

    return str(text).split()


def stopword_removal(tokens):

    stopwords = {
        "yang",
        "dan",
        "di",
        "ke",
        "dari",
        "ini",
        "itu",
        "untuk",
        "dengan",
        "pada",
        "adalah",
        "saya",
        "aku",
        "nya",
        "ga",
        "gak",
        "nggak",
        "tidak",
        "ada",
        "jadi",
        "karena",
        "atau",
        "juga",
        "sudah",
        "sangat",
        "lebih",
        "bisa",
        "dalam",
        "akan",
        "se",
        "aja"
    }

    return [
        word
        for word in tokens
        if word not in stopwords
    ]


def stemming(tokens):

    # Stemming disesuaikan dengan preprocessing
    # yang digunakan pada penelitian.

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

        st.metric(
            "Metode",
            "Naïve Bayes"
        )

    with col2:

        st.metric(
            "Jenis Analisis",
            "Sentimen"
        )

    with col3:

        st.metric(
            "Sumber Data",
            "Google Play Store"
        )

    st.info(
        """
        Aplikasi ini merupakan interface untuk membantu proses
        analisis sentimen ulasan pengguna aplikasi Blibli.
        """
    )

    st.markdown(
        "### Alur Penggunaan"
    )

    st.write(
        "1. Upload dataset ulasan."
    )

    st.write(
        "2. Periksa data."
    )

    st.write(
        "3. Jalankan preprocessing."
    )

    st.write(
        "4. Jalankan klasifikasi sentimen."
    )

    st.write(
        "5. Lihat hasil dan visualisasi."
    )

    st.write(
        "6. Download hasil analisis."
    )


# ============================================================
# UPLOAD DATASET
# ============================================================

elif menu == "Upload Dataset":

    st.subheader("📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=[
            "csv",
            "xlsx"
        ]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(
                ".csv"
            ):

                df = pd.read_csv(
                    uploaded_file
                )

            else:

                df = pd.read_excel(
                    uploaded_file
                )

            st.session_state.data = df

            st.session_state.hasil = None

            st.success(
                "✅ Dataset berhasil diunggah."
            )

            st.write(
                "### Informasi Dataset"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Jumlah Baris",
                    len(df)
                )

            with col2:

                st.metric(
                    "Jumlah Kolom",
                    len(df.columns)
                )

            st.write(
                "### Preview Dataset"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.write(
                "### Nama Kolom"
            )

            st.write(
                list(df.columns)
            )

        except Exception as e:

            st.error(
                f"Dataset gagal dibaca: {e}"
            )


# ============================================================
# PREPROCESSING
# ============================================================

elif menu == "Preprocessing":

    st.subheader(
        "🧹 Preprocessing Data"
    )

    if st.session_state.data is None:

        st.warning(
            "Silakan upload dataset terlebih dahulu."
        )

    else:

        df = st.session_state.data.copy()

        st.write(
            "### Pilih Kolom Ulasan"
        )

        text_column = st.selectbox(
            "Kolom teks ulasan:",
            df.columns
        )

        if st.button(
            "▶ Jalankan Preprocessing"
        ):

            with st.spinner(
                "Sedang melakukan preprocessing..."
            ):

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
                    .apply(
                        lambda x:
                        " ".join(x)
                    )
                )

                st.session_state.data = df

            st.success(
                "✅ Preprocessing berhasil."
            )

            st.write(
                "### Hasil Preprocessing"
            )

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

    st.subheader(
        "🤖 Analisis Sentimen Naïve Bayes"
    )

    if st.session_state.data is None:

        st.warning(
            "Silakan upload dataset terlebih dahulu."
        )

    else:

        df = st.session_state.data.copy()

        st.write(
            "### Upload Model Naïve Bayes"
        )

        uploaded_model = st.file_uploader(
            "Upload model Naïve Bayes (.pkl)",
            type=["pkl"],
            key="model_upload"
        )

        if uploaded_model is not None:

            try:

                model = pickle.load(
                    uploaded_model
                )

                st.success(
                    "✅ Model Naïve Bayes berhasil dimuat."
                )

                text_column = st.selectbox(
                    "Pilih kolom teks untuk prediksi:",
                    df.columns,
                    key="prediction_column"
                )

                if st.button(
                    "▶ Jalankan Analisis Sentimen"
                ):

                    with st.spinner(
                        "Model sedang melakukan prediksi..."
                    ):

                        teks = (
                            df[text_column]
                            .astype(str)
                        )

                        teks_bersih = (
                            teks
                            .apply(preprocess_text)
                        )

                        prediksi = model.predict(
                            teks_bersih
                        )

                        df["sentimen"] = prediksi

                        st.session_state.hasil = df

                    st.success(
                        "✅ Analisis sentimen berhasil dilakukan."
                    )

                    st.write(
                        "### Ringkasan Sentimen"
                    )

                    jumlah = (
                        df["sentimen"]
                        .value_counts()
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Positif",
                            int(
                                jumlah.get(
                                    "Positif",
                                    0
                                )
                            )
                        )

                    with col2:

                        st.metric(
                            "Netral",
                            int(
                                jumlah.get(
                                    "Netral",
                                    0
                                )
                            )
                        )

                    with col3:

                        st.metric(
                            "Negatif",
                            int(
                                jumlah.get(
                                    "Negatif",
                                    0
                                )
                            )
                        )

                    st.write(
                        "### Hasil Prediksi"
                    )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    csv = df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Hasil CSV",
                        data=csv,
                        file_name="hasil_sentimen_blibli.csv",
                        mime="text/csv"
                    )

            except Exception as e:

                st.error(
                    f"Model gagal digunakan: {e}"
                )


# ============================================================
# VISUALISASI
# ============================================================

elif menu == "Visualisasi":

    st.subheader(
        "📊 Visualisasi Hasil Sentimen"
    )

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
