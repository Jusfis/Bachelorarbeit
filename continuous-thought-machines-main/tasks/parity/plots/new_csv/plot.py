import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Set the seaborn theme for better aesthetics
sns.set_theme(style="darkgrid")

# Load the data
df_test_acc = pd.read_csv(os.path.join(BASE_DIR,"parity_10_5_test_accuracies.csv"))
df_test_loss = pd.read_csv(os.path.join(BASE_DIR,"parity_10_5_test_losses.csv"))
df_train_acc = pd.read_csv(os.path.join(BASE_DIR,"parity_10_5_train_accuracies.csv"))
df_train_loss = pd.read_csv(os.path.join(BASE_DIR,"parity_10_5_train_losses.csv"))


for col in df_train_loss.columns[1:]:  # Skip the 'Step' column, smooth everything else
    df_train_loss[col] = df_train_loss[col].rolling(window=100, min_periods=1).mean()

# Create a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(15, 10))


def plot_metric(df, ax, title, ylabel, metric_base_name):
    # Retrieve column names for KAN
    kan_col = f"postactivation_production: kan - {metric_base_name}"
    kan_min = f"{kan_col}__MIN"
    kan_max = f"{kan_col}__MAX"

    # Plot KAN main line and variance shadow
    sns.lineplot(data=df, x="Step", y=kan_col, ax=ax, label="KAN", color="blue")
    if kan_min in df.columns and kan_max in df.columns:
        ax.fill_between(df["Step"], df[kan_min], df[kan_max], color="blue", alpha=0.2)

    # Retrieve column names for MLP
    mlp_col = f"postactivation_production: mlp - {metric_base_name}"
    mlp_min = f"{mlp_col}__MIN"
    mlp_max = f"{mlp_col}__MAX"

    # Plot MLP main line and variance shadow
    sns.lineplot(data=df, x="Step", y=mlp_col, ax=ax, label="MLP", color="orange")
    if mlp_min in df.columns and mlp_max in df.columns:
        ax.fill_between(df["Step"], df[mlp_min], df[mlp_max], color="orange", alpha=0.2)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Step", fontsize=12)
    ax.legend()


# Plot all 4 metrics
plot_metric(df_train_loss, axes[0, 0], "Train Loss", "Loss", "Train/Losses_every_step")
plot_metric(df_test_loss, axes[0, 1], "Test Loss", "Loss", "Test/Losses")
plot_metric(df_train_acc, axes[1, 0], "Train Accuracy", "Accuracy", "Train/Accuracies_most_certain")
plot_metric(df_test_acc, axes[1, 1], "Test Accuracy", "Accuracy", "Test/Accuracies_most_certain")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR,"parity_10_5_metrics.png"), dpi=300)