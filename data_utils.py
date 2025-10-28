import os
import streamlit as st
import zipfile
import tempfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer


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
        df = pd.read_excel(uploaded_file)
        return _process_dataframe(df)

    # ---------------- ZIP Upload ----------------
    elif file_name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "data.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find files inside ZIP
            extracted_files = []
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    extracted_files.append(os.path.join(root, file))
            print("DEBUG – extracted files:", extracted_files)
            st.write("Detected files in ZIP:", extracted_files)

            # Case 1: ZIP of images (folders per class)
            if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in extracted_files):
                ds = tf.keras.utils.image_dataset_from_directory(
                    tmpdir,
                    image_size=(28, 28),
                    color_mode="grayscale",
                    batch_size=None
                )
                X = np.array([x.numpy() for x, _ in ds])
                X = X.reshape(X.shape[0], -1)
                y = np.array([y.numpy() for _, y in ds])
                y = y.flatten()
                class_names = ds.class_names
                return X, y, class_names

            # Case 2: ZIP of CSVs
            elif any(f.lower().endswith(".csv") for f in extracted_files):
                csvs = [pd.read_csv(f) for f in extracted_files if f.lower().endswith(".csv")]
                df = pd.concat(csvs, ignore_index=True)
                return _process_dataframe(df)

            else:
                raise ValueError(
                    "Unsupported ZIP format. Please upload a ZIP containing images or CSV files."
                )

    else:
        raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .zip file.")


def _process_dataframe(df):
    """Cleans tabular CSV/Excel and returns (X, y, class_names). Handles text columns."""
    df = df.dropna()

    # Detect target column
    y_col = None
    for col in df.columns:
        if col.lower() in ["label", "target", "class", "y", "status"]:
            y_col = col
            break
    if y_col is None:
        y_col = df.columns[-1]  # fallback to last column

    y = df[y_col]
    X = df.drop(columns=[y_col])

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = list(le.classes_)

    # Handle text columns
    text_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if text_cols:
        X_text = X[text_cols].astype(str).apply(lambda x: " ".join(x), axis=1)
        vectorizer = TfidfVectorizer(max_features=1000)
        X_vec = vectorizer.fit_transform(X_text).toarray()
        X = pd.DataFrame(X_vec)
    else:
        X = pd.get_dummies(X, drop_first=True)

    return X.values, y, class_names
