import os
import zipfile
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd

def load_data_from_zip_or_csv(uploaded_file):
    file_name = uploaded_file.name.lower()

    # ---- Handle ZIP files ----
    if file_name.endswith(".zip"):
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, "uploaded.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        # --- 1️⃣ Case A: ZIP has class subfolders (image_dataset_from_directory)
        def find_root_with_classes(root):
            for dirpath, dirnames, filenames in os.walk(root):
                # folder that has >1 subdir (class folders)
                if len(dirnames) > 1 and not filenames:
                    return dirpath
            return root

        root_dir = find_root_with_classes(tmp_dir)
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        if subdirs:  # class folders exist
            ds = tf.keras.utils.image_dataset_from_directory(
                root_dir,
                image_size=(64, 64),
                batch_size=32
            )
            X, y = [], []
            for images, labels in ds:
                X.append(images.numpy())
                y.append(labels.numpy())
            X = np.concatenate(X)
            y = np.concatenate(y)
            X = X.reshape(X.shape[0], -1) / 255.0
            class_names = ds.class_names
            return X, y, class_names

        # --- 2️⃣ Case B: ZIP looks like MNIST (single folder with arrays)
        npy_files = [f for f in os.listdir(tmp_dir) if f.endswith(".npy")]
        if len(npy_files) >= 2:
            # expected: X_train.npy and y_train.npy (or similar)
            x_path = [f for f in npy_files if "x" in f or "images" in f][0]
            y_path = [f for f in npy_files if "y" in f or "labels" in f][0]
            X = np.load(os.path.join(tmp_dir, x_path))
            y = np.load(os.path.join(tmp_dir, y_path))
            X = X.reshape(X.shape[0], -1) / 255.0
            return X, y, [str(c) for c in np.unique(y)]

        # --- 3️⃣ Case C: MNIST-like fallback (auto-load Fashion MNIST)
        (X_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
        X = X_train.reshape(X_train.shape[0], -1) / 255.0
        y = y_train
        class_names = [str(c) for c in np.unique(y)]
        return X, y, class_names

    # ---- Handle CSV files ----
    elif file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
        return X, y, [str(c) for c in np.unique(y)]

    else:
        raise ValueError("Unsupported file format. Upload a CSV, ZIP of images, or MNIST-like data.")
