import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf


def load_data_from_zip_or_csv(uploaded_file):
    """
    Handles CSV, Excel, or ZIP of images/tabular/text files.
    Returns X, y, class_names.
    """
    file_name = uploaded_file.name.lower()

    # ---------------- CSV Upload ----------------
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return _process_dataframe(df)

    # ---------------- EXCEL Upload ----------------
    elif file_name.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        except ImportError:
            raise ImportError("Please install 'openpyxl' to read Excel files (pip install openpyxl).")
        return _process_dataframe(df)

    # ---------------- ZIP Upload ----------------
    elif file_name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "data.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            extracted_files = []
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    extracted_files.append(os.path.join(root, file))

            st.write("Detected files in ZIP:", extracted_files)

            # --- Case 1: ZIP of images ---
            images = [f for f in extracted_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if images:
                ds = tf.keras.utils.image_dataset_from_directory(
                    tmpdir,
                    image_size=(28, 28),
                    color_mode="grayscale",
                    batch_size=None
                )
                X = np.array([x.numpy() for x, _ in ds])
                X = X.reshape(X.shape[0], -1)
                y = np.array([y.numpy() for _, y in ds]).flatten()
                class_names = ds.class_names
                return X, y, class_names

            # --- Case 2: ZIP of CSV files ---
            csvs = [f for f in extracted_files if f.lower().endswith(".csv")]
            if csvs:
                df_list = [pd.read_csv(f) for f in csvs]
                df = pd.concat(df_list, ignore_index=True)
                return _process_dataframe(df)

            # --- Case 3: ZIP of Excel files ---
            excels = [f for f in extracted_files if f.lower().endswith(".xlsx")]
            if excels:
                try:
                    with open(excels[0], "rb") as f:
                        df = pd.read_excel(f, engine="openpyxl")
                except ImportError:
                    raise ImportError("Please install 'openpyxl' to read Excel files (pip install openpyxl).")
                return _process_dataframe(df)

            # --- Unsupported content ---
            raise ValueError("Unsupported ZIP format. Please upload a ZIP containing images, CSVs, or Excel files.")

    else:
        raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .zip file.")


def _process_dataframe(df):
    """
    Cleans tabular CSV/Excel and returns (X, y, class_names).
    Handles text columns and fallback for label detection.
    """
    df = df.dropna()
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    # --- Detect label column ---
    y_col = None
    for col in df.columns:
        if col.lower() in ["label", "target", "class", "y", "status"]:
            y_col = col
            break

    if y_col is None:
        y_col = df.columns[-1]
        st.warning(f"No label column detected. Using the last column '{y_col}' as target.")

    # --- Separate X and y ---
    y = df[y_col]
    X = df.drop(columns=[y_col])

    # --- Encode target ---
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = list(le.classes_)

    # --- Handle text columns ---
    text_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if text_cols:
        X_text = X[text_cols].astype(str).apply(lambda x: " ".join(x), axis=1)
        vectorizer = TfidfVectorizer(max_features=1000)
        X_vec = vectorizer.fit_transform(X_text).toarray()
        X = pd.DataFrame(X_vec)
    else:
        X = pd.get_dummies(X, drop_first=True)

    # --- Return numpy arrays ---
    return X.values, y, class_names
