import pandas as pd
import numpy as np
import zipfile, tempfile, os
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_data_from_zip_or_csv(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        class_names = sorted(list(set(y)))
        return X, y, class_names

    elif uploaded_file.name.endswith(".zip"):
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        ds = image_dataset_from_directory(tmp_dir, image_size=(64, 64), batch_size=32)
        class_names = ds.class_names
        X, y = [], []
        for imgs, labels in ds:
            X.append(imgs.numpy())
            y.append(labels.numpy())
        X = np.concatenate(X)
        y = np.concatenate(y)
        X = X.reshape(len(X), -1) / 255.0  # flatten and normalize
        return X, y, class_names
