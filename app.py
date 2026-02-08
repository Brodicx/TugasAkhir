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

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Prediksi Stagnasi UMKM",
    layout="wide"
)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_kaggle_data():
    return pd.read_csv("Dataset/train/kaggle_umkm.csv")

@st.cache_data
def load_daerah_data():
    return {
        "Jawa Barat (Jenis)": "Dataset/gov_context/jawabaratjenis_umkm.csv",
        "Jawa Barat (Kota/Kab)": "Dataset/gov_context/jawabaratkotakab_umkm.csv",
        "Aceh": "Dataset/gov_context/aceh_umkm.csv",
        "Sukabumi": "Dataset/gov_context/sukabumi_umkm.csv",
        "Cirebon": "Dataset/gov_context/cirebon_umkm.csv",
        "Tasikmalaya": "Dataset/gov_context/tasikmalaya_umkm.csv"
    }

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_modeling_data(df):
    df = df.copy()

    num_cols = [
        "aset", "omset", "laba", "biaya_karyawan",
        "jumlah_pelanggan", "tenaga_kerja_perempuan",
        "tenaga_kerja_laki_laki", "kapasitas_produksi"
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    median_omset = df["omset"].median()
    median_laba = df["laba"].median()

    df["stagnasi"] = np.where(
        (df["omset"] < median_omset) & (df["laba"] < median_laba),
        1, 0
    )

    drop_cols = ["id_umkm", "nama_usaha", "tahun_berdiri"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = ["jenis_usaha", "marketplace", "status_legalitas"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop("stagnasi", axis=1)
    y = df["stagnasi"]

    return X, y

# ======================================================
# TRAIN MODEL
# ======================================================
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)

    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    return {
        "lr": lr,
        "xgb": xgb,
        "scaler": scaler,
        "X_columns": X.columns,
        "y_test": y_test,
        "lr_pred": lr_pred,
        "xgb_pred": xgb_pred,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape
    }

# ======================================================
# HEADER
# ======================================================
st.title("Prediksi Stagnasi UMKM Berbasis Machine Learning")
st.markdown(
    "Aplikasi ini membandingkan **Logistic Regression** dan **XGBoost** "
    "untuk memprediksi stagnasi UMKM menggunakan dataset Kaggle. "
    "Dataset UMKM daerah digunakan sebagai konteks dan EDA Indonesia."
)

tab1, tab2 = st.tabs([" Modeling & Prediksi", " EDA UMKM Daerah"])

# ======================================================
# TAB 1 : MODELING + INPUT MANUAL
# ======================================================
with tab1:
    df = load_kaggle_data()
    X, y = preprocess_modeling_data(df)
    model_pack = train_models(X, y)

    st.subheader("Informasi Data")
    st.write(f"Train: {model_pack['train_shape']}")
    st.write(f"Test: {model_pack['test_shape']}")
    st.write("Distribusi Label:")
    st.write(y.value_counts(normalize=True))

    metrics = pd.DataFrame({
        "Accuracy": [
            accuracy_score(model_pack["y_test"], model_pack["lr_pred"]),
            accuracy_score(model_pack["y_test"], model_pack["xgb_pred"])
        ],
        "Precision": [
            precision_score(model_pack["y_test"], model_pack["lr_pred"]),
            precision_score(model_pack["y_test"], model_pack["xgb_pred"])
        ],
        "Recall": [
            recall_score(model_pack["y_test"], model_pack["lr_pred"]),
            recall_score(model_pack["y_test"], model_pack["xgb_pred"])
        ],
        "F1-Score": [
            f1_score(model_pack["y_test"], model_pack["lr_pred"]),
            f1_score(model_pack["y_test"], model_pack["xgb_pred"])
        ]
    }, index=["Logistic Regression", "XGBoost"])

    st.subheader("Evaluasi Model")
    st.dataframe(metrics.style.format("{:.4f}"))

    st.subheader("Confusion Matrix")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(model_pack["y_test"], model_pack["lr_pred"])
        ).plot(ax=ax)
        ax.set_title("Logistic Regression")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(model_pack["y_test"], model_pack["xgb_pred"])
        ).plot(ax=ax)
        ax.set_title("XGBoost")
        st.pyplot(fig)

    # ===============================
    # INPUT MANUAL
    # ===============================
    st.subheader("Prediksi Stagnasi UMKM (Input Manual)")

    with st.form("form_prediksi"):
        jenis_usaha = st.selectbox(
            "Jenis Usaha", df["jenis_usaha"].unique()
        )
        marketplace = st.selectbox(
            "Menggunakan Marketplace", ["Ya", "Tidak"]
        )
        status_legalitas = st.selectbox(
            "Status Legalitas", df["status_legalitas"].unique()
        )
        aset = st.number_input("Aset", min_value=0.0)
        omset = st.number_input("Omset", min_value=0.0)
        laba = st.number_input("Laba", min_value=0.0)
        submit = st.form_submit_button("Prediksi")

    if submit:
        input_df = pd.DataFrame([{
            "aset": aset,
            "omset": omset,
            "laba": laba,
            "marketplace": marketplace,
            "jenis_usaha": jenis_usaha,
            "status_legalitas": status_legalitas
        }])

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(
            columns=model_pack["X_columns"], fill_value=0
        )

        input_scaled = model_pack["scaler"].transform(input_df)
        pred = model_pack["xgb"].predict(input_df)[0]

        if pred == 1:
            st.error("UMKM diprediksi mengalami stagnasi")
        else:
            st.success("UMKM diprediksi tidak mengalami stagnasi")

# ======================================================
# TAB 2 : EDA DAERAH
# ======================================================
with tab2:
    st.subheader("Eksplorasi UMKM Daerah")

    daerah_files = load_daerah_data()
    pilihan = st.selectbox(
        "Pilih Dataset Daerah", list(daerah_files.keys())
    )

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
