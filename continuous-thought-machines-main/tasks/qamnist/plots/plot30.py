import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_metrics_seaborn_smoothed(
        test_acc_file, test_loss_file, train_acc_file, train_loss_file,
        base_test_acc_file, base_test_loss_file, base_train_acc_file, base_train_loss_file
):
    # Load data
    df_test_acc = pd.read_csv(test_acc_file)
    df_test_loss = pd.read_csv(test_loss_file)
    df_train_acc = pd.read_csv(train_acc_file)
    df_train_loss = pd.read_csv(train_loss_file)

    df_baseline_test_acc = pd.read_csv(base_test_acc_file)
    df_baseline_test_loss = pd.read_csv(base_test_loss_file)
    df_baseline_train_acc = pd.read_csv(base_train_acc_file)
    df_baseline_train_loss = pd.read_csv(base_train_loss_file)

    # Subsample train data (every 20th step) to speed up plotting and reduce noise
    df_train_acc = df_train_acc.iloc[::20].copy()
    df_train_loss = df_train_loss.iloc[::20].copy()
    df_baseline_train_acc = df_baseline_train_acc.iloc[::20].copy()
    df_baseline_train_loss = df_baseline_train_loss.iloc[::20].copy()

    dfs = [
        df_test_acc, df_test_loss, df_train_acc, df_train_loss,
        df_baseline_test_acc, df_baseline_test_loss, df_baseline_train_acc, df_baseline_train_loss
    ]

    # 1. Convert all columns to numeric, sort by Step, AND filter up to 100k Steps
    for i in range(len(dfs)):
        for col in dfs[i].columns:
            dfs[i][col] = pd.to_numeric(dfs[i][col], errors='coerce')

        # Sort values
        dfs[i] = dfs[i].sort_values('Step')

        # LIMIT TO 100k ITERATIONS
        dfs[i] = dfs[i][dfs[i]['Step'] <= 100000]

        dfs[i] = dfs[i].reset_index(drop=True)

    # 2. Apply a sliding average to the Test Data
    window_size_test = 50
    for i in [0, 1, 4, 5]:  # Indices for test data
        for col in dfs[i].columns[1:]:  # Skip the 'Step' column
            dfs[i][col] = dfs[i][col].rolling(window=window_size_test, min_periods=1).mean()

    # 3. Apply a sliding average to the Training Data
    window_size_train = 50
    for i in [2, 3, 6, 7]:  # Indices for train data
        for col in dfs[i].columns[1:]:
            dfs[i][col] = dfs[i][col].rolling(window=window_size_train, min_periods=1).mean()

    # Unpack updated dataframes
    (df_test_acc, df_test_loss, df_train_acc, df_train_loss,
     df_baseline_test_acc, df_baseline_test_loss, df_baseline_train_acc, df_baseline_train_loss) = dfs

    sns.set_theme(style="darkgrid")
    colors = sns.color_palette("deep")

    # ==========================================
    # --- Plot 1 & 2: Test Accuracy & Loss ---
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    # KAN Test
    sns.lineplot(data=df_test_acc, x='Step', y=df_test_acc.iloc[:, 1], label='KAN Test Accuracy', color=colors[0],
                 ax=ax1)
    ax1.fill_between(df_test_acc['Step'], df_test_acc.iloc[:, 2], df_test_acc.iloc[:, 3], color=colors[0], alpha=0.2)

    # MLP Test
    sns.lineplot(data=df_test_acc, x='Step', y=df_test_acc.iloc[:, 4], label='MLP Test Accuracy', color=colors[2],
                 ax=ax1)
    ax1.fill_between(df_test_acc['Step'], df_test_acc.iloc[:, 5], df_test_acc.iloc[:, 6], color=colors[2], alpha=0.2)

    # Baseline Test Acc
    sns.lineplot(data=df_baseline_test_acc, x='Step', y=df_baseline_test_acc.iloc[:, 1], label='LSTM 1 Test Accuracy',
                 color=colors[3], ax=ax1)
    ax1.fill_between(df_baseline_test_acc['Step'], df_baseline_test_acc.iloc[:, 2], df_baseline_test_acc.iloc[:, 3],
                     color=colors[3], alpha=0.2)

    ax1.set_title('Test Accuracy (up to 100k)', fontsize=14)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    # Begrenze die x-Achse visuell nochmal zur Sicherheit
    ax1.set_xlim(0, 100000)

    # KAN Test Loss
    sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 1], label='KAN Test Loss', color=colors[0], ax=ax2)
    ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 2], df_test_loss.iloc[:, 3], color=colors[0], alpha=0.2)

    # MLP Test Loss
    sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 4], label='MLP Test Loss', color=colors[2], ax=ax2)
    ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 5], df_test_loss.iloc[:, 6], color=colors[2], alpha=0.2)

    # Baseline Test Loss
    sns.lineplot(data=df_baseline_test_loss, x='Step', y=df_baseline_test_loss.iloc[:, 1], label='LSTM 1 Test Loss',
                 color=colors[3], ax=ax2)
    ax2.fill_between(df_baseline_test_loss['Step'], df_baseline_test_loss.iloc[:, 2], df_baseline_test_loss.iloc[:, 3],
                     color=colors[3], alpha=0.2)

    ax2.set_title('Test Loss (up to 100k)', fontsize=14)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlim(0, 100000)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'test_qamnist_metrics_comparison_smoothed_30.png'), dpi=300)
    plt.show()

    # ==========================================
    # --- Plot 3 & 4: Train Accuracy & Loss ---
    # ==========================================
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # KAN Train Acc
    sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 1], label='KAN Train Accuracy', color=colors[0],
                 ax=ax3)
    ax3.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 2], df_train_acc.iloc[:, 3], color=colors[0], alpha=0.2)

    # MLP Train Acc
    sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 4], label='MLP Train Accuracy', color=colors[2],
                 ax=ax3)
    ax3.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 5], df_train_acc.iloc[:, 6], color=colors[2], alpha=0.2)

    # Baseline Train Acc
    sns.lineplot(data=df_baseline_train_acc, x='Step', y=df_baseline_train_acc.iloc[:, 1],
                 label='LSTM 1 Train Accuracy', color=colors[3], ax=ax3)
    ax3.fill_between(df_baseline_train_acc['Step'], df_baseline_train_acc.iloc[:, 2], df_baseline_train_acc.iloc[:, 3],
                     color=colors[3], alpha=0.2)

    ax3.set_title('Train Accuracy (up to 100k)', fontsize=14)
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_xlim(0, 100000)

    # KAN Train Loss
    sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 1], label='KAN Train Loss', color=colors[0],
                 ax=ax4)
    ax4.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 2], df_train_loss.iloc[:, 3], color=colors[0],
                     alpha=0.2)

    # MLP Train Loss
    sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 4], label='MLP Train Loss', color=colors[2],
                 ax=ax4)
    ax4.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 5], df_train_loss.iloc[:, 6], color=colors[2],
                     alpha=0.2)

    # Baseline Train Loss
    sns.lineplot(data=df_baseline_train_loss, x='Step', y=df_baseline_train_loss.iloc[:, 1], label='LSTM 1 Train Loss',
                 color=colors[3], ax=ax4)
    ax4.fill_between(df_baseline_train_loss['Step'], df_baseline_train_loss.iloc[:, 2],
                     df_baseline_train_loss.iloc[:, 3], color=colors[3], alpha=0.2)

    ax4.set_title('Train Loss (up to 100k)', fontsize=14)
    ax4.set_xlabel('Step', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_xlim(0, 100000)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'train_qamnist_metrics_comparison_smoothed_30.png'), dpi=300)
    plt.show()


# Run the function
plot_metrics_seaborn_smoothed(
    os.path.join(BASE_DIR, 'test_accuracies_qamnist_30.csv'),
    os.path.join(BASE_DIR, 'test_losses_qamnist_30.csv'),
    os.path.join(BASE_DIR, 'train_accuracies_qamnist_40.csv'),
    os.path.join(BASE_DIR, 'train_losses_qamnist_30.csv'),
    os.path.join(BASE_DIR, 'baseline_qamnist_test_accuracies.csv'),
    os.path.join(BASE_DIR, 'baseline_qamnist_test_losses.csv'),
    os.path.join(BASE_DIR, 'baseline_qamnist_train_accuracies.csv'),
    os.path.join(BASE_DIR, 'baseline_qamnist_train_losses.csv')
)