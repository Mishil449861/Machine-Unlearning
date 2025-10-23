import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import zipfile, os, tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom utility imports
from data_utils import load_data_from_zip_or_csv
from viz_utils import plot_probability_shift
from sisa_utils import (
    build_model,
    train_sisa_models,
    unlearn_class,
    get_ensemble_predictions
)



# Streamlit page setup
st.set_page_config(page_title="Machine Unlearning Lab", layout="wide")
st.title("Machine Unlearning Sandbox")
st.caption("Upload a dataset (CSV or ZIP), train SISA models, and visualize forgetting behavior.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Dataset (CSV or ZIP)", type=["csv", "zip"])

if uploaded_file:
    with st.spinner("Loading dataset..."):
        X, y, class_names = load_data_from_zip_or_csv(uploaded_file)

    st.success(f"Dataset loaded successfully with shape {X.shape}")
    st.write(f"Detected {len(class_names)} classes: {class_names}")

    # --- Split and Standardize ---
    X = StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train SISA Models ---
    num_shards = st.slider("Number of SISA Shards", 2, 10, 5)
    if st.button("Train SISA Models"):
        with st.spinner("Training shards..."):
            sisa_models, shards = train_sisa_models(x_train, y_train, num_shards)
        st.session_state.update({
            "models": sisa_models,
            "shards": shards,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "class_names": class_names
        })
        st.success("Training complete!")

# --- Unlearning and Visualization ---
if "models" in st.session_state:
    forget_class = st.selectbox("Select Class to Forget:", st.session_state["class_names"])

    if st.button("Forget and Visualize"):
        forget_idx = st.session_state["class_names"].index(forget_class)

        # ---- Predictions BEFORE unlearning ----
        preds_before = get_ensemble_predictions(
            st.session_state["models"],
            st.session_state["x_test"],
            return_proba=True
        )

        # ---- Perform unlearning ----
        with st.spinner(f"Forgetting class '{forget_class}'..."):
            models_after = unlearn_class(
                st.session_state["models"],
                st.session_state["shards"],
                st.session_state["x_train"],
                st.session_state["y_train"],
                forget_idx
            )

        # ---- Predictions AFTER unlearning ----
        preds_after = get_ensemble_predictions(
            models_after,
            st.session_state["x_test"],
            return_proba=True
        )

        # ---- Update models in session ----
        st.session_state["models"] = models_after

        # ---- Probability Shift Visualization ----
        st.subheader("Probability Shift After Forgetting")
        fig_shift = plot_probability_shift(preds_before, preds_after, st.session_state["class_names"])
        st.pyplot(fig_shift)
