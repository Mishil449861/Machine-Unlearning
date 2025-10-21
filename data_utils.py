import os
import zipfile
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd
from glob import glob

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

        # Find image files anywhere inside
        image_files = glob(os.path.join(tmp_dir, "**", "*.*"), recursive=True)
        image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]

        if not image_files:
            raise ValueError("No image files found in ZIP. Make sure it contains .jpg, .png, etc.")

        # Detect if there are class subfolders
        parent_dirs = [os.path.basename(os.path.dirname(f)) for f in image_files]
        unique_parents = list(set(parent_dirs))

        # If all images are in one folder → assign dummy class
        if len(unique_parents) == 1:
            root_dir = os.path.join(tmp_dir, "auto_class")
            os.makedirs(os.path.join(root_dir, "class0"), exist_ok=True)
            for img_path in image_files:
                os.rename(img_path, os.path.join(root_dir, "class0", os.path.basename(img_path)))
        else:
            # Move to new folder structure
            root_dir = os.path.join(tmp_dir, "organized")
            os.makedirs(root_dir, exist_ok=True)
            for f in image_files:
                label = os.path.basename(os.path.dirname(f))
                target_dir = os.path.join(root_dir, label)
                os.makedirs(target_dir, exist_ok=True)
                os.rename(f, os.path.join(target_dir, os.path.basename(f)))

        # Create dataset
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
        raise ValueError("Unsupported file format. Upload a CSV or ZIP containing images.")
