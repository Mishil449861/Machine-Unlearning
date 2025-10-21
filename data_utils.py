import os
import zipfile
import numpy as np
import tensorflow as tf
import pandas as pd
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

    # Case 2: ZIP dataset
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                files = os.listdir(tmpdirname)

                # ✅ Detect MNIST / Fashion-MNIST format (idx files)
                if any("idx" in f for f in files):
                    # Check which one it might be
                    if any("fashion" in f for f in files):
                        (X_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
                    else:
                        (X_train, y_train), _ = tf.keras.datasets.mnist.load_data()

                    # Flatten and normalize
                    X = X_train.reshape((X_train.shape[0], -1)).astype("float32") / 255.0
                    y = y_train
                    class_names = [str(i) for i in sorted(np.unique(y))]
                    return X, y, class_names

                # ✅ Otherwise, assume folder of images
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

                raise ValueError("No image files found in ZIP. Upload images or MNIST-style .idx files.")

    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .zip file.")
