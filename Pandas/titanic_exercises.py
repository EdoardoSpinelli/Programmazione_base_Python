# Titanic Dataset – Exercise Sheet with Solutions

"""
Questo file contiene una serie di esercizi (dal più semplice al più complesso) sul dataset
`titanic.csv` e le relative soluzioni, scritte in Python usando pandas, matplotlib e
scikit‑learn.  
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# 1️⃣ Caricamento e prima esplorazione: 
# Traccia:
# - Caricare il dataset Titanic da `titanic.csv` usando pandas.
# - Stampare le prime 5 righe del DataFrame.
# - Visualizzare le informazioni del DataFrame (tipo di colonna, valori nulli, ecc.).
# ---------------------------------------------------------------------------
print("--- 1️⃣ Caricamento e prima esplorazione ---")

df = pd.read_csv(
    r"C:/Users/francesca.laera/OneDrive - LUTECH SPA/Documenti/Desktop/ITS/Foggia 2025/LEZ 7/titanic.csv"
)
print(df.head())
print(df.info())

# ---------------------------------------------------------------------------
# 2️⃣ Statistiche descrittive di base
# Traccia:
# - Calcolare il numero totale di passeggeri.
# - Contare i sopravvissuti.
# - Calcolare età media, mediana e la classe più frequente.

# ---------------------------------------------------------------------------
print("\n--- 2️⃣ Statistiche descrittive di base ---")
print("Totale passeggeri:", len(df))
print("Sopravvissuti:", df["Survived"].sum())
print("Età media:", df["Age"].mean())
print("Età mediana:", df["Age"].median())
print("Classe più frequente:", df["Pclass"].mode()[0])

# ---------------------------------------------------------------------------
# 3️⃣ Filtri semplici
# Traccia:
# - Selezionare tutti i passeggeri maschi.
# - Selezionare le donne sopravvissute.
# - Estrarre le prime 10 persone con età > 30.

# ---------------------------------------------------------------------------
print("\n--- 3️⃣ Filtri semplici ---")
maschi = df[df["Sex"] == "male"]
print("Numero maschi:", len(maschi))

donne_sopravvissute = df[(df["Sex"] == "female") & (df["Survived"] == 1)]
print("Donne sopravvissute:", len(donne_sopravvissute))

over_30 = df[df["Age"] > 30].head(10)
print("Prime 10 persone con età > 30:\n", over_30)

# ---------------------------------------------------------------------------
# 4️⃣ Aggiungere e trasformare colonne
# Traccia:
# - Creare la colonna `FamilySize` (SibSp + Parch + 1).
# - Aggiungere la colonna booleana `IsAlone` (FamilySize == 1).
# - Normalizzare la colonna `Fare` con min‑max scaling.

# ---------------------------------------------------------------------------
print("\n--- 4️⃣ Aggiungere e trasformare colonne ---")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = df["FamilySize"] == 1
df["Fare_norm"] = (df["Fare"] - df["Fare"].min()) / (df["Fare"].max() - df["Fare"].min())
print(df[["FamilySize", "IsAlone", "Fare_norm"].head())

# ---------------------------------------------------------------------------
# 5️⃣ Raggruppamenti e aggregazioni
# Traccia:
# - Calcolare il tasso di sopravvivenza per classe.
# - Calcolare l’età media per combinazione di sesso e classe.
# - Identificare il porto di imbarco con più sopravvissuti.

# ---------------------------------------------------------------------------
print("\n--- 5️⃣ Raggruppamenti e aggregazioni ---")
survival_by_class = df.groupby("Pclass")["Survived"].mean()
print("Tasso sopravvivenza per classe:\n", survival_by_class)

age_by_sex_class = df.groupby(["Sex", "Pclass"])["Age"].mean()
print("Età media per Sex e Pclass:\n", age_by_sex_class)

top_embarked = df.groupby("Embarked")["Survived"].sum().idxmax()
print("Porto di imbarco con più sopravvissuti:", top_embarked)

# ---------------------------------------------------------------------------
# 6️⃣ Operazioni di pivot e visualizzazione rapida
# Traccia:
# - Creare una pivot‑table per contare i passeggeri per sesso e porto.
# - Disegnare un istogramma dell’età dei sopravvissuti.

# ---------------------------------------------------------------------------
print("\n--- 6️⃣ Pivot e visualizzazioni ---")
pivot = df.pivot_table(
    index="Sex",
    columns="Embarked",
    values="PassengerId",
    aggfunc="count",
    fill_value=0,
)
print("Pivot (numero passeggeri per Sex e Embarked):\n", pivot)

# Istogramma età dei sopravvissuti
plt.figure(figsize=(8, 4))
df[df["Survived"] == 1]["Age"].hist(bins=20, edgecolor="black")
plt.title("Distribuzione età dei sopravvissuti")
plt.xlabel("Età")
plt.ylabel("Frequenza")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 7️⃣ Modello predittivo di base (Logistic Regression)
# Traccia:
# - Preparare i dati (rimuovere colonne non numeriche, gestire missing values).
# - Suddividere in train/test (80/20).
# - Addestrare una regressione logistica.
# - Stampare l’accuratezza sul set di test.

# ---------------------------------------------------------------------------
print("\n--- 7️⃣ Modello predittivo (Logistic Regression) ---")
# Preparazione dati
X = df.drop(columns=["Survived", "Name", "Ticket", "Cabin"])
y = df["Survived"]

numeric_features = ["Age", "Fare", "FamilySize"]
categorical_features = ["Sex", "Embarked", "Pclass"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(steps=[("preprocess", preprocess), ("clf", LogisticRegression(max_iter=200))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuratezza sul test set:", accuracy_score(y_test, y_pred))

