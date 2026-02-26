import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def aggregate_seeds(file_list):


    dfs = []
    for file in file_list:
        df = pd.read_csv(file)
        # Wir gehen davon aus, dass 'Step' existiert und der Wert in der 2. Spalte steht
        val_col = df.columns[1]
        # Nur Step und den Wert behalten, Wert-Spalte einheitlich benennen
        df_clean = df[['Step', val_col]].rename(columns={val_col: 'Value'})
        dfs.append(df_clean)

    # Alle DataFrames untereinanderhängen
    combined_df = pd.concat(dfs)

    # Nach Step gruppieren und Mean, Min, Max berechnen
    agg_df = combined_df.groupby('Step')['Value'].agg(['mean', 'min', 'max']).reset_index()
    return agg_df


def plot_metrics_seaborn(test_acc_files, test_loss_files, train_acc_files, train_loss_files):
    # Daten aggregieren
    df_test_acc = aggregate_seeds(test_acc_files)
    df_test_loss = aggregate_seeds(test_loss_files)
    df_train_acc = aggregate_seeds(train_acc_files)
    df_train_loss = aggregate_seeds(train_loss_files)

    sns.set_theme(style="darkgrid")
    colors = sns.color_palette("deep")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Accuracy ---
    # Linien (Mittelwert)
    sns.lineplot(data=df_train_acc, x='Step', y='mean', label='Train Accuracy', color=colors[0], ax=ax1)
    sns.lineplot(data=df_test_acc, x='Step', y='mean', label='Test Accuracy', color=colors[1], ax=ax1)

    # Schattierte Fläche (Min bis Max)
    ax1.fill_between(df_train_acc['Step'], df_train_acc['min'], df_train_acc['max'], color=colors[0], alpha=0.2)
    ax1.fill_between(df_test_acc['Step'], df_test_acc['min'], df_test_acc['max'], color=colors[1], alpha=0.2)

    ax1.set_title('Accuracy over steps', fontsize=14)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)

    # --- Plot 2: Loss ---
    # Linien (Mittelwert)
    sns.lineplot(data=df_train_loss, x='Step', y='mean', label='Train Loss', color=colors[0], ax=ax2)
    sns.lineplot(data=df_test_loss, x='Step', y='mean', label='Test Loss', color=colors[1], ax=ax2)

    # Schattierte Fläche (Min bis Max)
    ax2.fill_between(df_train_loss['Step'], df_train_loss['min'], df_train_loss['max'], color=colors[0], alpha=0.2)
    ax2.fill_between(df_test_loss['Step'], df_test_loss['min'], df_test_loss['max'], color=colors[1], alpha=0.2)

    ax2.set_title('Loss over steps', fontsize=14)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'listops_loss_accuracies_mlp.png'), dpi=300)
    plt.show()


# --- Ausführung ---

# Listen mit den Dateinamen für die 3 Seeds erstellen
test_acc_files = [os.path.join(BASE_DIR, f) for f in
                  ['6_test_accuracies_mlp.csv', '30_test_accuracies_mlp.csv', '49_test_accuracies_mlp.csv']]
test_loss_files = [os.path.join(BASE_DIR, f) for f in
                   ['6_test_losses_mlp.csv', '30_test_losses_mlp.csv', '49_test_losses_mlp.csv']]
train_acc_files = [os.path.join(BASE_DIR, f) for f in
                   ['6_train_accuracies_mlp.csv', '30_train_accuracies_mlp.csv', '49_train_accuracies_mlp.csv']]
train_loss_files = [os.path.join(BASE_DIR, f) for f in
                    ['6_train_losses_mlp.csv', '30_train_losses_mlp.csv', '49_train_losses_mlp.csv']]

# Funktion aufrufen
plot_metrics_seaborn(test_acc_files, test_loss_files, train_acc_files, train_loss_files)