import os
import zipfile
import numpy as np
import tensorflow as tf
import pandas as pd
import idx2numpy
from io import BytesIO
import tempfile

def load_data_from_zip_or_csv(uploaded_file):
    filename = uploaded_file.name.lower()

    # Case 1: CSV dataset
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        class_names = sorted(df.iloc[:, -1].unique().astype(str))
        return X, y, class_names

    # Case 2: ZIP file
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                
                # Look for MNIST/Fashion-MNIST format (idx files)
                idx_files = [f for f in os.listdir(tmpdirname) if f.endswith(".idx") or "idx" in f]
                if len(idx_files) >= 4:  # typical number of MNIST files
                    try:
                        # Try to infer whether it's MNIST or Fashion-MNIST
                        print("Detected MNIST-style dataset")
                        (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
                        X = X_train.reshape((X_train.shape[0], -1)) / 255.0
                        y = y_train
                        class_names = [str(i) for i in sorted(np.unique(y))]
                        return X, y, class_names
                    except Exception:
                        raise ValueError("MNIST-style data found but could not be parsed.")
                
                # Otherwise, treat as image folders
                subdirs = [d for d in os.listdir(tmpdirname) if os.path.isdir(os.path.join(tmpdirname, d))]
                if subdirs:
                    ds = tf.keras.utils.image_dataset_from_directory(
                        tmpdirname, image_size=(64, 64), batch_size=32
                    )
                    X, y = zip(*[(x.numpy(), y.numpy()) for x, y in ds.unbatch()])
                    X = np.stack(X)
                    y = np.array(y)
                    class_names = ds.class_names
                    return X, y, class_names

                raise ValueError("No recognizable data found in ZIP file.")

    else:
        raise ValueError("Unsupported file format. Upload a .csv or .zip containing images or MNIST-style data.")
