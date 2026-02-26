import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Alle CSV-Dateien auslesen
files = [f for f in os.listdir('.') if f.endswith('.csv') and 'kan' in f]

data_frames = []

for file in files:
    parts = file.replace('.csv', '').split('_')
    metric_type = parts[2].capitalize()  # 'Test' oder 'Train'
    metric_name = parts[3].capitalize()  # 'Accuracies' oder 'Losses'

    df = pd.read_csv(file)
    step_col = df.columns[0]
    val_col = df.columns[1]  # Spalte mit den Hauptwerten

    # Einheitlichen DataFrame erstellen (Run-ID wird hier weggelassen,
    # damit seaborn automatisch über die mehrfachen Werte pro "Step" aggregiert)
    temp_df = pd.DataFrame({
        'Step': df[step_col],
        'Value': df[val_col],
        'Phase': metric_type,
        'Metric': metric_name
    })
    data_frames.append(temp_df)

all_data = pd.concat(data_frames, ignore_index=True)

sns.set_theme(style="darkgrid")

# ================================
# PLOT 1: Accuracies
# ================================
plt.figure(figsize=(8, 5))
acc_data = all_data[all_data['Metric'] == 'Accuracies']
# errorbar=('ci', 95) ist der Standard in aktuellen seaborn-Versionen
sns.lineplot(data=acc_data, x='Step', y='Value', hue='Phase', errorbar=('ci', 95))
plt.title('Accuracies (Train vs Test) with 95% Confidence Interval')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('accuracies_plot.png', dpi=300)
plt.close()

# ================================
# PLOT 2: Losses
# ================================
plt.figure(figsize=(8, 5))
loss_data = all_data[all_data['Metric'] == 'Losses']
sns.lineplot(data=loss_data, x='Step', y='Value', hue='Phase', errorbar=('ci', 95))
plt.title('Losses (Train vs Test) with 95% Confidence Interval')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('losses_plot.png', dpi=300)
plt.close()

# Wenn du das Script lokal bei dir ohne speichern anzeigen willst,
# kannst du die plt.savefig() Zeilen durch plt.show() ersetzen.