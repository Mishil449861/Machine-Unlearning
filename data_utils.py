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

        # Try to find directory with subfolders (class dirs)
        def find_root_with_classes(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if len(dirnames) > 1 and not filenames:
                    return dirpath
            return root  # fallback

        root_dir = find_root_with_classes(tmp_dir)

        # Check that we have at least one subfolder
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        if not class_dirs:
            raise ValueError(
                "ZIP must contain subfolders (one per class). "
                "Example: cats/, dogs/, etc."
            )

        ds = tf.keras.utils.image_dataset_from_directory(
            root_dir,
            image_size=(64, 64),
            batch_size=32
        )

        X, y = [], []
        class_names = ds.class_names

        for images, labels in ds:
            X.append(images.numpy())
            y.append(labels.numpy())

        X = np.concatenate(X)
        y = np.concatenate(y)
        X = X.reshape(X.shape[0], -1) / 255.0  # flatten and normalize

        return X, y, class_names

    # ---- Handle CSV files ----
    elif file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
        return X, y, [str(c) for c in np.unique(y)]

    else:
        raise ValueError("Unsupported file format. Please upload a CSV or a ZIP of images.")
