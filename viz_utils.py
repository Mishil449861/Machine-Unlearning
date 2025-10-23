if st.button("Forget and Visualize"):
    forget_idx = st.session_state["class_names"].index(forget_class)

    # Store predictions BEFORE unlearning
    preds_before = get_ensemble_predictions(
        st.session_state["models"],
        st.session_state["x_test"],
        return_proba=True
    )

    # Perform unlearning
    models_after = unlearn_class(
        st.session_state["models"],
        st.session_state["shards"],
        st.session_state["x_train"],
        st.session_state["y_train"],
        forget_idx
    )

    # Predictions AFTER unlearning
    preds_after = get_ensemble_predictions(
        models_after,
        st.session_state["x_test"],
        return_proba=True
    )

    # Update session
    st.session_state["models"] = models_after

    # Visualize
    st.subheader("Probability Shift After Forgetting")
    fig_shift = plot_probability_shift(preds_before, preds_after, st.session_state["class_names"])
    st.pyplot(fig_shift)

    st.subheader("Confusion Matrix After Forgetting")
    y_pred = np.argmax(preds_after, axis=1)
    fig_cm = plot_confusion_matrix(st.session_state["y_test"], y_pred, st.session_state["class_names"])
    st.pyplot(fig_cm)
