
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_metrics_seaborn(test_acc_file, test_loss_file, train_acc_file, train_loss_file):

    df_test_acc = pd.read_csv(test_acc_file)
    df_test_loss = pd.read_csv(test_loss_file)
    df_train_acc = pd.read_csv(train_acc_file)
    df_train_loss = pd.read_csv(train_loss_file)


    sns.set_theme(style="darkgrid")
    colors = sns.color_palette("deep")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Accuracy ---
    sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 1], label='Train Accuracy', color=colors[0],
                 ax=ax1)
    sns.lineplot(data=df_test_acc, x='Step', y=df_test_acc.iloc[:, 1], label='Test Accuracy', color=colors[1], ax=ax1)


    ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 2], df_train_acc.iloc[:, 3], color=colors[0], alpha=0.2)
    ax1.fill_between(df_test_acc['Step'], df_test_acc.iloc[:, 2], df_test_acc.iloc[:, 3], color=colors[1], alpha=0.2)

    ax1.set_title('Accuracy over steps', fontsize=14)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)

    # --- Plot 2: Loss ---
    sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 1], label='Train Loss', color=colors[0], ax=ax2)
    sns.lineplot(data=df_test_loss, x='Step', y=df_test_loss.iloc[:, 1], label='Test Loss', color=colors[1], ax=ax2)


    ax2.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 2], df_train_loss.iloc[:, 3], color=colors[0],
                     alpha=0.2)
    ax2.fill_between(df_test_loss['Step'], df_test_loss.iloc[:, 2], df_test_loss.iloc[:, 3], color=colors[1], alpha=0.2)

    ax2.set_title('Loss over steps', fontsize=14)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR,'listops_loss_75_25_accuracies.png' ),dpi=300)
    plt.show()






plot_metrics_seaborn(
    os.path.join(BASE_DIR, 'test_acc_kan_listops.csv'),
    os.path.join(BASE_DIR, 'test_Losses_kan_listops.csv'),
    os.path.join(BASE_DIR, 'train_acc_kan_listops.csv'),
    os.path.join(BASE_DIR, 'train_losses_kan_listops.csv')
)