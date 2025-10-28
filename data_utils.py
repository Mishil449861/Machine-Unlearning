import os
import io
import zipfile
import pandas as pd
import numpy as np
from PIL import Image

def load_data_from_zip_or_csv(uploaded_file):
    """
    Load dataset from a Streamlit-uploaded file.
    Supports:
      - CSV
      - Excel (XLSX)
      - ZIP (images in subfolders OR CSV/Excel inside ZIP)
    """
    def _process_dataframe(df):
        # auto-detect label column if present
        label_col = None
        for col in df.columns:
            if col.lower() in ["label", "class", "target", "y"]:
                label_col = col
                break
        if label_col is None:
            raise ValueError("No label column found in dataset.")
        X = df.drop(columns=[label_col]).values
        y = df[label_col].values
        class_names = sorted(df[label_col].unique().tolist())
        return X, y, class_names

    # --- handle CSV ---
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return _process_dataframe(df)

    # --- handle Excel ---
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        return _process_dataframe(df)

    # --- handle ZIP ---
    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            # Get all file names
            names = [n for n in zf.namelist() if not n.endswith("/")]
            csvs = [n for n in names if n.lower().endswith(".csv")]
            excels = [n for n in names if n.lower().endswith((".xlsx", ".xls"))]
            images = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]

            # --- case 1: ZIP contains CSV ---
            if csvs:
                with zf.open(csvs[0]) as f:
                    df = pd.read_csv(f)
                return _process_dataframe(df)

            # --- case 2: ZIP contains Excel ---
            if excels:
                with zf.open(excels[0]) as f:
                    df = pd.read_excel(f)
                return _process_dataframe(df)

            # --- case 3: ZIP contains image folders ---
            if images:
                from sklearn.preprocessing import LabelEncoder
                img_data, labels = [], []
                for name in images:
                    # assume folder structure class_name/image.jpg
                    parts = name.split("/")
                    if len(parts) < 2:
                        continue
                    label = parts[-2]
                    with zf.open(name) as f:
                        img = Image.open(f).convert("RGB").resize((64, 64))
                        img_data.append(np.array(img).flatten())
                        labels.append(label)
                if not img_data:
                    raise ValueError("No valid images found in ZIP.")
                X = np.array(img_data)
                le = LabelEncoder()
                y = le.fit_transform(labels)
                return X, y, le.classes_.tolist()

            # --- fallback ---
            raise ValueError(
                "Unsupported ZIP format. ZIP must contain either CSV, Excel, or images in subfolders."
            )

    # --- unsupported file ---
    raise ValueError("Unsupported file type. Please upload a CSV, Excel, or ZIP file.")
