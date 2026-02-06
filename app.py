import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

st.set_page_config(
    page_title="Prediksi Stagnasi UMKM",
    layout="wide"
)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_kaggle_data():
    df = pd.read_csv("Dataset/train/kaggle_umkm.csv")
    return df

@st.cache_data
def load_daerah_data():
    files = {
        "Jawa Barat (Jenis)": "Dataset/gov_context/jawabaratjenis_umkm.csv",
        "Jawa Barat (Kota/Kab)": "Dataset/gov_context/jawabaratkotakab_umkm.csv",
        "Aceh": "Dataset/gov_context/aceh_umkm.csv",
        "Sukabumi": "Dataset/gov_context/sukabumi_umkm.csv",
        "Cirebon": "Dataset/gov_context/cirebon_umkm.csv",
        "Tasikmalaya": "Dataset/gov_context/tasikmalaya_umkm.csv"
    }
    return files

# ======================================================
# PREPROCESSING & LABEL
# ======================================================
def preprocess_modeling_data(df):
    df = df.copy()

    # kolom numerik
    num_cols = [
        "aset", "omset", "laba", "biaya_karyawan",
        "jumlah_pelanggan", "tenaga_kerja_perempuan",
        "tenaga_kerja_laki_laki", "kapasitas_produksi"
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # label stagnasi
    median_omset = df["omset"].median()
    median_laba = df["laba"].median()

    df["stagnasi"] = np.where(
        (df["omset"] < median_omset) & (df["laba"] < median_laba),
        1, 0
    )

    # drop kolom non-prediktor
    drop_cols = ["id_umkm", "nama_usaha", "tahun_berdiri"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # encoding kategorikal
    cat_cols = ["jenis_usaha", "marketplace", "status_legalitas"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop("stagnasi", axis=1)
    y = df["stagnasi"]

    return X, y

# ======================================================
# MODELING
# ======================================================
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)

    # XGBoost
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    results = {
        "Logistic Regression": {
            "model": lr,
            "y_pred": lr_pred,
            "scaled": True
        },
        "XGBoost": {
            "model": xgb,
            "y_pred": xgb_pred,
            "scaled": False
        }
    }

    return results, y_test, X_train.shape, X_test.shape

# ======================================================
# UI
# ======================================================
st.title("📊 Prediksi Stagnasi UMKM Berbasis Machine Learning")
st.markdown("""
Aplikasi ini membandingkan **Logistic Regression** dan **XGBoost**
untuk memprediksi stagnasi UMKM menggunakan **dataset Kaggle**.
Dataset UMKM daerah digunakan sebagai **EDA dan konteks Indonesia**.
""")

tab1, tab2 = st.tabs(["📈 Modeling", "🌍 EDA UMKM Daerah"])

# ======================================================
# TAB 1 : MODELING
# ======================================================
with tab1:
    df = load_kaggle_data()
    X, y = preprocess_modeling_data(df)

    results, y_test, train_shape, test_shape = train_models(X, y)

    st.subheader("📦 Informasi Data")
    st.write(f"Train: {train_shape}")
    st.write(f"Test: {test_shape}")
    st.write("Distribusi Label:")
    st.write(y.value_counts(normalize=True))

    metrics = []

    for name, res in results.items():
        y_pred = res["y_pred"]
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        })

    df_metrics = pd.DataFrame(metrics).set_index("Model")
    st.subheader("📊 Evaluasi Model")
    st.dataframe(df_metrics.style.format("{:.4f}"))

    st.subheader("📉 Confusion Matrix")
    col1, col2 = st.columns(2)

    for col, (name, res) in zip([col1, col2], results.items()):
        with col:
            cm = confusion_matrix(y_test, res["y_pred"])
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=ax)
            ax.set_title(name)
            st.pyplot(fig)

# ======================================================
# TAB 2 : EDA UMKM DAERAH
# ======================================================
with tab2:
    st.subheader("📍 Eksplorasi UMKM Daerah")

    daerah_files = load_daerah_data()
    pilihan = st.selectbox("Pilih Dataset Daerah", list(daerah_files.keys()))

    df_daerah = pd.read_csv(daerah_files[pilihan])
    st.dataframe(df_daerah.head())

    if "jenis_usaha" in df_daerah.columns:
        st.subheader("Distribusi Jenis Usaha")
        fig, ax = plt.subplots()
        df_daerah["jenis_usaha"].value_counts().head(10).plot(
            kind="bar", ax=ax
        )
        ax.set_ylabel("Jumlah UMKM")
        ax.set_title(f"Jenis UMKM - {pilihan}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
