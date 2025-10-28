import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import struct


def read_idx(filename):
    """Reads IDX formatted binary files (used in MNIST/Fashion-MNIST)."""
    with open(filename, "rb") as f:
        zero, data_type, dims = struct.unpack(">HBB", f.read(4))
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_data_from_zip_or_csv(uploaded_file):
    """
    Handles CSV, Excel, IDX, or ZIP containing images/tabular/text files.
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

            # --- Case 4: ZIP of IDX files (MNIST/Fashion-MNIST) ---
            idx_images = [f for f in extracted_files if "images" in f.lower() and f.lower().endswith(".idx3-ubyte")]
            idx_labels = [f for f in extracted_files if "labels" in f.lower() and f.lower().endswith(".idx1-ubyte")]

            if idx_images and idx_labels:
                X = read_idx(idx_images[0])
                y = read_idx(idx_labels[0])

                # Flatten and normalize images
                X = X.reshape(X.shape[0], -1) / 255.0
                y = y.flatten()

                # Default class names for Fashion-MNIST
                if "fashion" in file_name or "fmnist" in file_name:
                    class_names = [
                        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                    ]
                else:
                    class_names = [str(i) for i in range(10)]

                return X, y, class_names

            # --- Unsupported content ---
            raise ValueError("Unsupported ZIP format. Please upload a ZIP containing images, CSVs, Excels, or IDX files.")

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

    return X.values, y, class_names
