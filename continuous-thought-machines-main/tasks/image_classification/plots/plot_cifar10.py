import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the seaborn theme for better aesthetics
sns.set_theme(style="darkgrid")
colors = sns.color_palette("deep")

# Load the data
df_test_loss_base = pd.read_csv(os.path.join(BASE_DIR, "resnet_cifar_test_losses.csv"))
df_test_acc_base = pd.read_csv(os.path.join(BASE_DIR, "resnet_cifar_test_accuracies.csv"))
df_train_loss_base = pd.read_csv(os.path.join(BASE_DIR, "resnet_cifar_train_losses.csv"))
df_train_acc_base = pd.read_csv(os.path.join(BASE_DIR, "resnet_cifar_train_accuracies.csv"))




# Smoothing der Trainingsdaten (Trainings-Loss & Train-Accuracy)
for col in df_train_loss_base.columns[1:]:  # Skip the 'Step' column
    df_train_loss_base[col] = df_train_loss_base[col].rolling(window=10, min_periods=1).mean()

for col in df_train_acc_base.columns[1:]:
    df_train_acc_base[col] = df_train_acc_base[col].rolling(window=10, min_periods=1).mean()

df_train_acc_base = df_train_acc_base[df_train_acc_base['Step'] <= 150000]
df_train_loss_base = df_train_loss_base[df_train_loss_base['Step'] <= 150000]
df_test_acc_base = df_test_acc_base[df_test_acc_base['Step'] <= 150000]
df_test_loss_base = df_test_loss_base[df_test_loss_base['Step'] <= 150000]

fig, axes = plt.subplots(2, 1, figsize=(15, 10))



def plot_metric_base(df, ax, title, ylabel, metric_base_name, color=colors[1], labelInput = None):
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
plot_metric_base(df_train_loss_base, axes[0], "Losses Resnet", "Loss",
            "model_type: resnet - Train/Losses", color=colors[0], labelInput="Train Losses")

plot_metric_base(df_test_loss_base, axes[0], "Losses Resnet", "Loss",
            "model_type: resnet - Test/Losses", color=colors[1], labelInput="Test Losses")

plot_metric_base(df_train_acc_base, axes[1], "Accuracies Resnet", "Accuracy",
            "model_type: resnet - Train/Accuracies", color=colors[0], labelInput="Train Accuracies")

plot_metric_base(df_test_acc_base, axes[1], "Accuracies Resnet", "Accuracy",
            "model_type: resnet - Test/Accuracies", color=colors[1], labelInput="Test Accuracies")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "resnet_cifar_metrics_new.png"), dpi=300)

# plot imagenet 25 10


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14, 6))
# TODO compute test accuracy from test loss later
df_train_acc = pd.read_csv(os.path.join(BASE_DIR, "train_accuracies_imagenet_25_10.csv"))
df_train_loss = pd.read_csv(os.path.join(BASE_DIR, "train_losses_imagenet_25_10.csv"))
df_test_loss = pd.read_csv(os.path.join(BASE_DIR, "test_losses_imagenet_25_10.csv"))

# Smoothing der Trainingsdaten (Trainings-Loss & Train-Accuracy)

for col in df_train_acc.columns[1:]:
    df_train_acc[col] = df_train_acc[col].rolling(window=20, min_periods=1).mean()

for col in df_train_loss.columns[1:]:
    df_train_loss[col] = df_train_loss[col].rolling(window=20, min_periods=1).mean()

# x min max

sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 1], label=f'KAN Train Accuracy ', color=colors[0], ax=ax1)
ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 2], df_train_acc.iloc[:, 3], color=colors[0], alpha=0.2)

sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 4], label=f'MLP Train Accuracy ', color=colors[1], ax=ax1)
ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 5], df_train_acc.iloc[:, 6], color=colors[1], alpha=0.2)

sns.lineplot(data=df_train_acc_base, x='Step', y=df_train_acc_base.iloc[:, 1], label=f'Baseline Train Accuracy ', color=colors[2], ax=ax1)
ax1.fill_between(df_train_acc_base['Step'], df_train_acc_base.iloc[:, 2], df_train_acc_base.iloc[:, 3], color=colors[2], alpha=0.2)



sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 1], label=f'KAN Train Loss ', color=colors[0], ax=ax2)
ax2.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 2], df_train_loss.iloc[:, 3], color=colors[0], alpha=0.2)

sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 4], label=f'MLP Train Loss ', color=colors[1], ax=ax2)
ax2.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 5], df_train_loss.iloc[:, 6], color=colors[1], alpha=0.2)


sns.lineplot(data=df_train_loss_base, x='Step', y=df_train_loss_base.iloc[:, 1], label=f'Baseline Train Loss ', color=colors[2], ax=ax2)
ax2.fill_between(df_train_loss_base['Step'], df_train_loss_base.iloc[:, 2], df_train_loss_base.iloc[:, 3], color=colors[2], alpha=0.2)




ax1.set_title('Train Accuracy', fontsize=14)
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)

ax2.set_title('Train Loss', fontsize=14)
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "train_cifar_25_10.png"), dpi=300)
plt.show()

df_test_loss = pd.read_csv(os.path.join(BASE_DIR, "test_losses_imagenet_25_10.csv"))

# Smoothing der Trainingsdaten (Trainings-Loss & Train-Accuracy)
#
for col in df_test_loss.columns[1:]:
    df_test_loss[col] = df_test_loss[col].rolling(window=2, min_periods=1).mean()

fig, (ax2) = plt.subplots(1, 1, figsize=(14, 6))



# sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 1], label=f'KAN Train Accuracy ', color=colors[0], ax=ax1)
# ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 2], df_train_acc.iloc[:, 3], color=colors[0], alpha=0.2)
#
# sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 4], label=f'MLP Train Accuracy ', color=colors[1], ax=ax1)
# ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 5], df_train_acc.iloc[:, 6], color=colors[1], alpha=0.2)


sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 1], label=f'KAN Test Loss ', color=colors[0], ax=ax2)
ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 2], df_test_loss.iloc[:, 3], color=colors[0], alpha=0.2)

sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 4], label=f'MLP Test Loss ', color=colors[1], ax=ax2)
ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 5], df_test_loss.iloc[:, 6], color=colors[1], alpha=0.2)

sns.lineplot(data=df_test_loss_base, x='Step', y=df_test_loss_base.iloc[:, 1], label=f'Baseline Test Loss ', color=colors[2], ax=ax2)
ax2.fill_between(df_test_loss_base['Step'], df_test_loss_base.iloc[:, 2], df_test_loss_base.iloc[:, 3], color=colors[2], alpha=0.2)


ax2.set_title('Test Loss', fontsize=14)
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "test_cifar_25_10.png"), dpi=300)
plt.show()
