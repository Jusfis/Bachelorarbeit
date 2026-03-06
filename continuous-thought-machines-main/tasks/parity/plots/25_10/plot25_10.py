import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the seaborn theme for better aesthetics
sns.set_theme(style="darkgrid")
colors = sns.color_palette("deep")

# Load the data (hier gehen wir davon aus, dass die anderen Dateien im selben Ordner liegen)
df_test_loss = pd.read_csv(os.path.join(BASE_DIR, "test_losses_parity_25_10.csv"))
df_test_acc = pd.read_csv(os.path.join(BASE_DIR, "test_accuracies_parity_25_10.csv"))
df_train_loss = pd.read_csv(os.path.join(BASE_DIR, "train_losses_parity_25_10.csv"))
df_train_acc = pd.read_csv(os.path.join(BASE_DIR, "train_accuracies_parity_25_10.csv"))

# BASELINE DATEN LADEN (Tippfehler bei 'final.csv' behoben)
df_train_loss_base = pd.read_csv(os.path.join(BASE_DIR, "baseline_3_parit_train_every_bit.csv"))
df_train_acc_base = pd.read_csv(os.path.join(BASE_DIR, "baseline_3_parity_accuracies_every_bit.csv"))

df_train_loss_base = df_train_loss_base[df_train_loss_base['Step'] <= 150000]
df_train_acc_base = df_train_acc_base[df_train_acc_base['Step'] <= 150000]


# Smoothing der Trainingsdaten (Trainings-Loss & Train-Accuracy)
for col in df_train_loss.columns[1:]:  # Skip the 'Step' column
    df_train_loss[col] = pd.to_numeric(df_train_loss[col], errors='coerce')
    df_train_loss[col] = df_train_loss[col].rolling(window=10, min_periods=1).mean()

for col in df_train_acc.columns[1:]:
    df_train_acc[col] = pd.to_numeric(df_train_acc[col], errors='coerce')
    df_train_acc[col] = df_train_acc[col].rolling(window=10, min_periods=1).mean()

for col in df_train_acc_base.columns[1:]:
    df_train_acc_base[col] = pd.to_numeric(df_train_acc_base[col], errors='coerce')
    df_train_acc_base[col] = df_train_acc_base[col].rolling(window=10, min_periods=1).mean()

for col in df_train_loss_base.columns[1:]:  # Skip the 'Step' column
    df_train_loss_base[col] = pd.to_numeric(df_train_loss_base[col], errors='coerce')
    df_train_loss_base[col] = df_train_loss_base[col].rolling(window=10, min_periods=1).mean()


def aggregate_baseline(df):
    """Berechnet den Durchschnitt der Metriken, MIN und MAX über alle Baseline-Runs."""
    main_cols = [col for col in df.columns if col != 'Step' and '__MIN' not in col and '__MAX' not in col]
    min_cols = [col for col in df.columns if '__MIN' in col]
    max_cols = [col for col in df.columns if '__MAX' in col]

    # Neue gemittelte Spalten anlegen
    df['Aggregated_Baseline'] = df[main_cols].mean(axis=1)
    df['Aggregated_Baseline__MIN'] = df[min_cols].mean(axis=1)
    df['Aggregated_Baseline__MAX'] = df[max_cols].mean(axis=1)

    return df

# Baseline aggregieren
df_train_loss_base = aggregate_baseline(df_train_loss_base)
df_train_acc_base = aggregate_baseline(df_train_acc_base)


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 6))


# --- Plot 1: Accuracy (ax1) ---
sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 1], label=f'KAN Train Accuracy ', color=colors[0], ax=ax1)
ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 2], df_train_acc.iloc[:, 3], color=colors[0], alpha=0.2)

sns.lineplot(data=df_train_acc, x='Step', y=df_train_acc.iloc[:, 4], label=f'MLP Train Accuracy ', color=colors[1], ax=ax1)
ax1.fill_between(df_train_acc['Step'], df_train_acc.iloc[:, 5], df_train_acc.iloc[:, 6], color=colors[1], alpha=0.2)

# HIER ANGEPASST: Nutze die aggregierten Spalten statt iloc[:, 1]
sns.lineplot(data=df_train_acc_base, x='Step', y='Aggregated_Baseline', label=f'Baseline Train Accuracy ', color=colors[2], ax=ax1)
ax1.fill_between(df_train_acc_base['Step'], df_train_acc_base['Aggregated_Baseline__MIN'], df_train_acc_base['Aggregated_Baseline__MAX'], color=colors[2], alpha=0.2)


# --- Plot 2: Loss (ax2) ---
sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 1], label=f'KAN Train Losses ', color=colors[0], ax=ax2)
ax2.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 2], df_train_loss.iloc[:, 3], color=colors[0], alpha=0.2)

sns.lineplot(data=df_train_loss, x='Step', y=df_train_loss.iloc[:, 4], label=f'MLP Train Losses ', color=colors[1], ax=ax2)
ax2.fill_between(df_train_loss['Step'], df_train_loss.iloc[:, 5], df_train_loss.iloc[:, 6], color=colors[1], alpha=0.2)

# HIER ANGEPASST: Nutze die aggregierten Spalten statt iloc[:, 1]
sns.lineplot(data=df_train_loss_base, x='Step', y='Aggregated_Baseline', label=f'Baseline Train Loss ', color=colors[2], ax=ax2)
ax2.fill_between(df_train_loss_base['Step'], df_train_loss_base['Aggregated_Baseline__MIN'], df_train_loss_base['Aggregated_Baseline__MAX'], color=colors[2], alpha=0.2)


ax1.set_title('Train Accuracy', fontsize=14)
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)

ax2.set_title('Train Loss', fontsize=14)
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "train_parity_25_10.png"), dpi=300)
# plt.show() # Auskommentiert für automatisiertes Speichern, kann bei Bedarf wieder rein