def plot_probability_shift(preds_before, preds_after, class_names):
    mean_before = np.mean(preds_before, axis=0)
    mean_after = np.mean(preds_after, axis=0)
    diff = mean_after - mean_before
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_names, diff)
    ax.set_title("Probability Shift per Class (After Forgetting)")
    ax.set_ylabel("Δ Probability (After - Before)")
    ax.axhline(0, color='gray', linestyle='--')
    plt.xticks(rotation=45)
    return fig
