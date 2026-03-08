import wandb
import pandas as pd

api = wandb.Api()

# 1. Lade die Runs aus deinem Projekt/Sweep
sweep_id = "numcr0oq"  # Ersetze durch deine Sweep-ID
runs = api.runs("justus-fischer-ludwig-maximilian-university-of-munich/ctm-parity",
                filters={"sweep": sweep_id})

all_histories = []
print(f"Lade Daten von {len(runs)} Runs...")

for run in runs:
    # Lade die Historie (samples hoch setzen, damit nichts fehlt!)
    hist = run.history(samples=50000)

    # Füge hinzu, ob es KAN oder MLP ist
    hist['Model_Type'] = run.config.get('postactivation_production', 'unknown')
    all_histories.append(hist)

# 2. Alles zu einem großen DataFrame zusammenfügen
df_all = pd.concat(all_histories, ignore_index=True)

# 3. WICHTIG: Das Step-Mismatch Problem beheben!
# Wir runden die Steps z.B. auf die nächsten 100. (101 wird zu 100, 199 wird zu 100)
# Passe die '100' an deine Logging-Frequenz (track_every) an!
# Wenn du alle 1000 Steps trackst, trage hier 1000 ein.
df_all['Step_Rounded'] = (df_all['_step'] // 100) * 100

# 4. Die Metrik wählen, die aggregiert werden soll
metric_name = "Train/Accuracies_every_step"  # An deinen W&B Namen anpassen

# 5. Gruppieren nach dem GERUNDETEN Step und dem Modell
# Jetzt fallen alle Runs am gleichen (gerundeten) Zeitpunkt zusammen!
df_aggregated = df_all.groupby(['Model_Type', 'Step_Rounded'])[metric_name].agg(
    MIN='min',
    MAX='max',
    MEAN='mean'
).reset_index()

# 6. Umbenennen, damit es zu deinem Plot-Skript passt
df_aggregated = df_aggregated.rename(columns={'Step_Rounded': 'Step'})

# 7. Als CSV speichern
df_aggregated.to_csv("Train_Accuracies_Aggregated.csv", index=False)
print("CSV erfolgreich mit echten Min/Max Werten exportiert!")