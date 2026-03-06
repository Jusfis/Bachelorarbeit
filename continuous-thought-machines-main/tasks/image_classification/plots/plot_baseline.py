import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the seaborn theme for better aesthetics
sns.set_theme(style="darkgrid")

# Load the data
df_test_loss = pd.read_csv(os.path.join(BASE_DIR, "resnet_cifar_test_losses.csv"))
df_test_acc = pd.read_csv(os.path.join(BASE_DIR, "resnet_cifar_test_accuracies.csv"))
df_train_loss = pd.read_csv(os.path.join(BASE_DIR, "resnet_cifar_train_losses.csv"))
df_train_acc = pd.read_csv(os.path.join(BASE_DIR, "resnet_cifar_train_accuracies.csv"))

# Smoothing der Trainingsdaten (Trainings-Loss & Train-Accuracy)
for col in df_train_loss.columns[1:]:  # Skip the 'Step' column
    df_train_loss[col] = df_train_loss[col].rolling(window=10, min_periods=1).mean()

for col in df_train_acc.columns[1:]:
    df_train_acc[col] = df_train_acc[col].rolling(window=10, min_periods=1).mean()

df_train_acc = df_train_acc[df_train_acc['Step'] <= 50000]
df_train_loss = df_train_loss[df_train_loss['Step'] <= 50000]
df_test_acc = df_test_acc[df_test_acc['Step'] <= 50000]
df_test_loss = df_test_loss[df_test_loss['Step'] <= 50000]

fig, axes = plt.subplots(2, 1, figsize=(15, 10))


def plot_metric(df, ax, title, ylabel, metric_base_name, color="blue", labelInput = None):
    # Retrieve column names for the dataset
    main_col = metric_base_name
    min_col = f"{metric_base_name}__MIN"
    max_col = f"{metric_base_name}__MAX"

    # Plot main line and variance shadow
    sns.lineplot(data=df, x="Step", y=main_col, ax=ax, label=labelInput, color=color)

    # Fill between if MIN and MAX columns exist
    if min_col in df.columns and max_col in df.columns:
        ax.fill_between(df["Step"], df[min_col], df[max_col], color=color, alpha=0.2)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Step", fontsize=12)
    ax.legend(loc="best")


# Plot all 4 metrics
plot_metric(df_train_loss, axes[0], "Losses Resnet", "Loss",
            "model_type: resnet - Train/Losses", color="blue", labelInput="Train Losses")

plot_metric(df_test_loss, axes[0], "Losses Resnet", "Loss",
            "model_type: resnet - Test/Losses", color="red",labelInput="Test Losses")

plot_metric(df_train_acc, axes[1], "Accuracies Resnet", "Accuracy",
            "model_type: resnet - Train/Accuracies", color="blue",labelInput="Train Accuracies")

plot_metric(df_test_acc, axes[1], "Accuracies Resnet", "Accuracy",
            "model_type: resnet - Test/Accuracies", color="red",labelInput="Test Accuracies")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "resnet_cifar_metrics.png"), dpi=300)