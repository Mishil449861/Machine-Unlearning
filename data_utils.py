import os
import zipfile
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd
from glob import glob

def load_data_from_zip_or_csv(uploaded_file):
    file_name = uploaded_file.name.lower()

    # ---- Handle Keras or NumPy MNIST files ----
    if file_name.endswith(".npz"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        data = np.load(tmp_path)
        # Try standard keys
        if "x_train" in data and "y_train" in data:
            X = data["x_train"]
            y = data["y_train"]
        elif "arr_0" in data and "arr_1" in data:
            X = data["arr_0"]
            y = data["arr_1"]
        else:
            raise ValueError("Unknown NPZ structure.")
        if X.ndim == 3:  # (n, 28, 28)
            X = X.reshape(X.shape[0], -1)
        X = X / 255.0
        class_names = [str(i) for i in np.unique(y)]
        return X, y, class_names

    elif file_name.endswith(".gz") and "mnist" in file_name:
        # Automatically load MNIST from TensorFlow if user uploaded raw files
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        X = x_train.reshape(x_train.shape[0], -1) / 255.0
        y = y_train
        class_names = [str(i) for i in np.unique(y)]
        return X, y, class_names

    # ---- Handle ZIP files ----
    elif file_name.endswith(".zip"):
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Find image files anywhere inside
        image_files = glob(os.path.join(tmp_dir, "**", "*.*"), recursive=True)
        image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        if not image_files:
            raise ValueError("No image files found in ZIP. Must contain .jpg, .png, etc.")

        parent_dirs = [os.path.basename(os.path.dirname(f)) for f in image_files]
        unique_parents = list(set(parent_dirs))

        if len(unique_parents) == 1:
            root_dir = os.path.join(tmp_dir, "auto_class")
            os.makedirs(os.path.join(root_dir, "class0"), exist_ok=True)
            for img_path in image_files:
                os.rename(img_path, os.path.join(root_dir, "class0", os.path.basename(img_path)))
        else:
            root_dir = os.path.join(tmp_dir, "organized")
            os.makedirs(root_dir, exist_ok=True)
            for f in image_files:
                label = os.path.basename(os.path.dirname(f))
                target_dir = os.path.join(root_dir, label)
                os.makedirs(target_dir, exist_ok=True)
                os.rename(f, os.path.join(target_dir, os.path.basename(f)))

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

    # ---- Handle CSV files ----
    elif file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
        return X, y, [str(c) for c in np.unique(y)]

    else:
        raise ValueError(
            "Unsupported file format. Upload one of: CSV, ZIP (images), NPZ, or MNIST GZ."
        )
