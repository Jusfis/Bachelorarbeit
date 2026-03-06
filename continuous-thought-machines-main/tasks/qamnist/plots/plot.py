import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_metrics_seaborn_smoothed(test_acc_file, test_loss_file, train_acc_file, train_loss_file):
    # Load data
    # df_test_acc = pd.read_csv(test_acc_file)
    # df_test_loss = pd.read_csv(test_loss_file)
    # df_train_acc = pd.read_csv(train_acc_file)
    # df_train_loss = pd.read_csv(train_loss_file)

    df_test_acc = pd.read_csv(os.path.join(BASE_DIR, 'final_qamnist_3_1_test_accuracies.csv'))
    df_test_loss = pd.read_csv(os.path.join(BASE_DIR, 'final_qamnist_3_1_test_losses.csv'))
    df_train_acc = pd.read_csv(os.path.join(BASE_DIR, 'final_qamnist_3_1_train_accuracies.csv'))
    df_train_loss = pd.read_csv(os.path.join(BASE_DIR, 'final_qamnist_3_1_train_losses.csv'))
    df_baseline_train_acc = pd.read_csv(os.path.join(BASE_DIR, 'baseline_qamnist_train_accuracies.csv'))
    df_baseline_train_loss = pd.read_csv(os.path.join(BASE_DIR, 'baseline_qamnist_train_losses.csv'))
    df_baseline_test_acc = pd.read_csv(os.path.join(BASE_DIR, 'baseline_qamnist_test_accuracies.csv'))
    df_baseline_test_loss = pd.read_csv(os.path.join(BASE_DIR, 'baseline_qamnist_test_losses.csv'))

    df_train_acc = df_train_acc[::20]
    df_train_loss = df_train_loss[::20]

    dfs = [df_test_acc, df_test_loss, df_train_acc, df_train_loss,df_baseline_train_acc, df_baseline_train_loss,df_baseline_test_acc, df_baseline_test_loss]

    # 1. Convert all columns to numeric and sort by Step chronologically
    for i in range(len(dfs)):
        for col in dfs[i].columns:
            dfs[i][col] = pd.to_numeric(dfs[i][col], errors='coerce')
        dfs[i] = dfs[i].sort_values('Step').reset_index(drop=True)




    # 2. Apply a sliding average to the Test Data (Accuracy and Loss)
    window_size = 50
    for i in [0,1,4,5,6,7]:  # Index 2 is train_acc, Index 3 is train_loss and baseline
        for col in dfs[i].columns[1:]:  # Skip the 'Step' column, smooth everything else
            dfs[i][col] = dfs[i][col].rolling(window=window_size, min_periods=1).mean()

    #     # 2. Apply a sliding average to the Training Data (Accuracy and Loss)
    window_size = 50
    for i in [ 2, 3]:  # Index 2 is train_acc, Index 3 is train_loss
        for col in dfs[i].columns[1:]:  # Skip the 'Step' column, smooth everything else
            dfs[i][col] = dfs[i][col].rolling(window=window_size, min_periods=1).mean()



    # Unpack updated dataframes
    df_test_acc, df_test_loss, df_train_acc, df_train_loss,df_baseline_train_acc, df_baseline_train_loss,df_baseline_test_acc, df_baseline_test_loss = dfs

    sns.set_theme(style="darkgrid")
    colors = sns.color_palette("deep")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))



    # --- Plot 1: Accuracy ---

    # KAN Test with final csv
    sns.lineplot(data=df_test_acc, x='Step', y=df_test_acc.iloc[:, 1], label=f'KAN Test Accuracy ', color=colors[0], ax=ax1)
    ax1.fill_between(df_test_acc['Step'], df_test_acc.iloc[:, 2], df_test_acc.iloc[:, 3], color=colors[0], alpha=0.2)

    #Baseline accuracy Test
    sns.lineplot(data=df_baseline_test_acc, x='Step', y=df_baseline_test_acc.iloc[:, 1], label=f'LSTM 1 Test Accuracy', color=colors[3], ax=ax1)
    ax1.fill_between(df_baseline_test_acc['Step'], df_baseline_test_acc.iloc[:, 2], df_baseline_test_acc.iloc[:, 3], color=colors[3], alpha=0.2)

    # MLP Test with final csv
    sns.lineplot(data=df_test_acc, x='Step', y=df_test_acc.iloc[:, 4], label=f'MLP Test Accuracy   ', color=colors[2], ax=ax1)
    ax1.fill_between(df_test_acc['Step'], df_test_acc.iloc[:, 5], df_test_acc.iloc[:, 6], color=colors[2], alpha=0.2)

    ax1.set_title('Test Accuracy', fontsize=14)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)



    # --- Plot 2: Loss ---

    # KAN Test with final csv
    sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 1], label=f'KAN Test Loss ', color=colors[0], ax=ax2)
    ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 2], df_test_loss.iloc[:, 3], color=colors[0], alpha=0.2)


    # MLP Test with final csv
    sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 4], label=f'MLP Test Loss ', color=colors[2], ax=ax2)
    ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 5], df_test_loss.iloc[:, 6], color=colors[2], alpha=0.2)

    # Baseline Test Loss
    sns.lineplot(data=df_baseline_test_loss, x='Step', y=df_baseline_test_loss.iloc[:, 1], label=f'LSTM 1 Test Loss', color=colors[3], ax=ax2)
    ax2.fill_between(df_baseline_test_loss['Step'], df_baseline_test_loss.iloc[:, 2], df_baseline_test_loss.iloc[:, 3], color=colors[3], alpha=0.2)

    ax2.set_title('Test Loss', fontsize=14)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)



    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'test_qamnist_metrics_comparison_smoothed_final_with_baseline.png'), dpi=300)
    plt.show()

    # next, we will plot the smoothed training metrics separately for better visibility

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))


    # KAN Train in final (Smoothed)
    sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 1], label=f'MLP Train Accuracy',
                 color=colors[0], ax=ax3)
    ax3.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 2], df_train_acc.iloc[:, 3], color=colors[0], alpha=0.2)

    # MLP Train in final (Smoothed)
    sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 4], label=f'MLP Train Accuracy',
                 color=colors[2], ax=ax3)
    ax3.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 5], df_train_acc.iloc[:, 6], color=colors[2], alpha=0.2)

    # Baseline accuracy train
    sns.lineplot(data=df_baseline_train_acc, x='Step', y=df_baseline_train_acc.iloc[:, 1], label=f'LSTM 1 Train Accuracy',
                 color=colors[3], ax=ax3)
    ax3.fill_between(df_baseline_train_acc['Step'], df_baseline_train_acc.iloc[:, 2], df_baseline_train_acc.iloc[:, 3],
                     color=colors[3], alpha=0.2)

    ax3.set_title('Train Accuracy', fontsize=14)
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)



    # KAN Train final (Smoothed)
    sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 1], label=f'KAN Train Loss',
                 color=colors[0], ax=ax4)
    ax4.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 2], df_train_loss.iloc[:, 3], color=colors[0],
                     alpha=0.2)

    # MLP Train final (Smoothed)
    sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 4], label=f'MLP Train Loss',
                 color=colors[2], ax=ax4)
    ax4.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 5], df_train_loss.iloc[:, 6], color=colors[2],
                     alpha=0.2)

    # Baseline Loss train
    sns.lineplot(data=df_baseline_train_loss, x='Step', y=df_baseline_train_loss.iloc[:, 1],
                 label=f'LSTM 1 Train Loss',
                 color=colors[3], ax=ax4)
    ax4.fill_between(df_baseline_train_loss['Step'], df_baseline_train_loss.iloc[:, 2], df_baseline_train_loss.iloc[:, 3],
                     color=colors[3], alpha=0.2)

    ax4.set_title('Train Loss', fontsize=14)
    ax4.set_xlabel('Step', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'train_qamnist_metrics_comparison_smoothed_final_with_baseline.png'), dpi=300)
    plt.show()

# Run the function
plot_metrics_seaborn_smoothed(
    os.path.join(BASE_DIR, 'qamnist_3_1_test_accuracies_kan.csv'),
    os.path.join(BASE_DIR, 'qamnist_3_1_test_losses_kan.csv'),
    os.path.join(BASE_DIR, 'qamnist_3_1_train_accuracies_kan.csv'),
    os.path.join(BASE_DIR, 'qamnist_3_1_train_losses_kan.csv')
)