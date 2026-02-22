import random
import numpy as np
import os

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
END = "]"

# Hier legst du fest, welche Operatoren vorkommen dürfen
OPERATORS = [MIN, MAX, MED, SUM_MOD]
VALUES = range(10)

# ==========================================
# ⚙️ EINSTELLUNGEN FÜR DEIN DATASET
# ==========================================
OUTPUT_FILE = "tasks/listops/dataset/train_easy_200_000.tsv"  # Hier wird die Datei gespeichert
DATA_POINTS = 200000
MAX_DEPTH = 3
MAX_ARGS = 3
VALUE_P = 0.25  # Wahrscheinlichkeit für eine reine Zahl statt Operator


# ==========================================


def generate_tree(depth):
    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1))

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t


def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'


def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return (np.sum(l[1]) % 10)
    elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])


if __name__ == "__main__":
    # Ordner erstellen, falls er noch nicht existiert
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)

    print(f"Generiere {DATA_POINTS} Aufgaben (Max Tiefe: {MAX_DEPTH}, Max Args: {MAX_ARGS})...")

    data = set()
    while len(data) < DATA_POINTS:
        data.add(generate_tree(1))

    print(f"Speichere in {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Header für Pandas
        f.write("Target\tSource\n")

        for example in data:
            target = str(to_value(example))
            source = to_string(example)
            # Schreibe Zeile für Zeile in die TSV
            f.write(f"{target}\t{source}\n")

    print("Fertig! Du kannst die Datei jetzt in dein Training laden.")