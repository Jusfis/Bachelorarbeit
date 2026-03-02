import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_metrics_seaborn_smoothed(test_acc_file, test_loss_file, train_acc_file, train_loss_file):
    # Load data
    df_test_acc = pd.read_csv(test_acc_file)
    df_test_loss = pd.read_csv(test_loss_file)
    df_train_acc = pd.read_csv(train_acc_file)
    df_train_loss = pd.read_csv(train_loss_file)

    dfs = [df_test_acc, df_test_loss, df_train_acc, df_train_loss]

    # 1. Convert all columns to numeric and sort by Step chronologically
    for i in range(len(dfs)):
        for col in dfs[i].columns:
            dfs[i][col] = pd.to_numeric(dfs[i][col], errors='coerce')
        dfs[i] = dfs[i].sort_values('Step').reset_index(drop=True)

    # 2. Apply a sliding average to the Training Data (Accuracy and Loss)
    window_size = 50
    for i in [0,1,2, 3]:  # Index 2 is train_acc, Index 3 is train_loss
        for col in dfs[i].columns[1:]:  # Skip the 'Step' column, smooth everything else
            dfs[i][col] = dfs[i][col].rolling(window=window_size, min_periods=1).mean()

    # Unpack updated dataframes
    df_test_acc, df_test_loss, df_train_acc, df_train_loss = dfs

    sns.set_theme(style="darkgrid")
    colors = sns.color_palette("deep")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Accuracy ---
    # # MLP Train (Smoothed)
    # sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 1], label=f'MLP Train Acc (MA={window_size})',
    #              color=colors[0], ax=ax1)
    # ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 2], df_train_acc.iloc[:, 3], color=colors[0], alpha=0.2)

    # MLP Test
    sns.lineplot(data=df_test_acc, x='Step', y=df_test_acc.iloc[:, 1], label=f'MLP Test Acc (MA={window_size})', color=colors[1], ax=ax1)
    ax1.fill_between(df_test_acc['Step'], df_test_acc.iloc[:, 2], df_test_acc.iloc[:, 3], color=colors[1], alpha=0.2)
    #
    # # KAN Train (Smoothed)
    # sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 4], label=f'KAN Train Acc (MA={window_size})',
    #              color=colors[2], ax=ax1)
    # ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 5], df_train_acc.iloc[:, 6], color=colors[2], alpha=0.2)

    # KAN Test
    sns.lineplot(data=df_test_acc, x='Step', y=df_test_acc.iloc[:, 4], label=f'KAN Test Acc   (MA={window_size})', color=colors[3], ax=ax1)
    ax1.fill_between(df_test_acc['Step'], df_test_acc.iloc[:, 5], df_test_acc.iloc[:, 6], color=colors[3], alpha=0.2)

    ax1.set_title('Accuracy over steps', fontsize=14)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)

    # --- Plot 2: Loss ---
    # # MLP Train (Smoothed)
    # sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 1], label=f'MLP Train Loss (MA={window_size})',
    #              color=colors[0], ax=ax2)
    # ax2.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 2], df_train_loss.iloc[:, 3], color=colors[0],
    #                  alpha=0.2)

    # MLP Test
    sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 1], label=f'MLP Test Loss (MA={window_size})', color=colors[1], ax=ax2)
    ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 2], df_test_loss.iloc[:, 3], color=colors[1], alpha=0.2)

    # # KAN Train (Smoothed)
    # sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 4], label=f'KAN Train Loss (MA={window_size})',
    #              color=colors[2], ax=ax2)
    # ax2.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 5], df_train_loss.iloc[:, 6], color=colors[2],
    #                  alpha=0.2)

    # KAN Test
    sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 4], label=f'KAN Test Loss (MA={window_size})', color=colors[3], ax=ax2)
    ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 5], df_test_loss.iloc[:, 6], color=colors[3], alpha=0.2)

    ax2.set_title('Loss over steps', fontsize=14)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'qamnist_metrics_comparison_smoothed.png'), dpi=300)
    plt.show()


# Run the function
plot_metrics_seaborn_smoothed(
    os.path.join(BASE_DIR, 'qamnist_3_1_test_accuracies_kan.csv'),
    os.path.join(BASE_DIR, 'qamnist_3_1_test_losses_kan.csv'),
    os.path.join(BASE_DIR, 'qamnist_3_1_train_accuracies_kan.csv'),
    os.path.join(BASE_DIR, 'qamnist_3_1_train_losses_kan.csv')
)