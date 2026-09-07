
# === SHAP Beeswarm Utility ===
def generate_shap_beeswarm(model, input_df, top_features_txt):
    import shap
    import matplotlib.pyplot as plt

    # Load top 20 SHAP feature names
    with open(top_features_txt, "r") as f:
        top_features = [line.strip() for line in f.readlines()]

    # Subset the input features to match top-k
    X = input_df[top_features]

    # Explain prediction for the single segment
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Pick class-specific values if multiclass
    if isinstance(shap_values, list):  # multiclass: shap_values[class][sample, feature]
        shap_values = shap_values[0]   # just show class 0 as approximation

    # Plot beeswarm
    shap.summary_plot(shap_values, X, plot_type="dot", max_display=5, show=False)

    # Save temporary plot and load into Streamlit
    fig_path = "temp_shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return fig_path
