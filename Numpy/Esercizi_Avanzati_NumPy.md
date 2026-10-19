# Esercizi Avanzati NumPy (Estensioni)

---

## 1. Fancy Indexing e Maschere Complesse
**Obiettivo:**
1. Creare una matrice 5x5 di numeri interi da 0 a 24.
2. Selezionare le righe 1, 3 e 4 e le colonne 0, 2, 4 usando *fancy indexing*.
3. Generare una maschera booleana che selezioni tutti gli elementi **pari** maggiori di 10.
4. Usare la maschera per estrarre i valori corrispondenti.

```python
import numpy as np

# 1. Matrice 5x5
A = np.arange(25).reshape(5, 5)
print("Matrice A:\n", A)

# 2. Fancy indexing (righe 1,3,4; colonne 0,2,4)
rows = [1, 3, 4]
cols = [0, 2, 4]
sub = A[rows][:, cols]
print("Sotto‑matrice con fancy indexing:\n", sub)

# 3. Maschera per valori pari > 10
mask = (A % 2 == 0) & (A > 10)
print("Maschera (pari e >10):\n", mask)

# 4. Estrarre gli elementi con la maschera
filtered = A[mask]
print("Elementi filtrati:\n", filtered)
```

---

## 2. Structured Arrays (Array di Record)
**Obiettivo:**
1. Definire un dtype con i campi `nome` (stringa 10), `età` (int), `altezza` (float).
2. Creare un array di 4 record.
3. Estrarre tutti i nomi delle persone con età > 20.

```python
import numpy as np

# 1. Definizione dtype
person_dtype = np.dtype([('nome', 'U10'), ('età', 'i4'), ('altezza', 'f4')])

# 2. Array di record
people = np.array([
    ('Anna', 22, 1.68),
    ('Luca', 19, 1.75),
    ('Mara', 25, 1.60),
    ('Giov', 23, 1.80)
], dtype=person_dtype)
print("Array di record:\n", people)

# 3. Filtrare per età > 20 e stampare i nomi
mask = people['età'] > 20
nomi = people['nome'][mask]
print("Nomi con età >20:", nomi)
```

---

## 3. Algebra Lineare: Moltiplicazione, Inversa, Autovalori
**Obiettivo:**
1. Creare due matrici 3x3 `A` e `B` casuali (interi 1‑9).
2. Calcolare il prodotto matriciale `C = A @ B`.
3. Calcolare l'inversa di `A` (se invertibile).
4. Ottenere gli autovalori di `C`.

```python
import numpy as np

# 1. Matrici casuali
rng = np.random.default_rng(seed=42)
A = rng.integers(1, 10, size=(3, 3))
B = rng.integers(1, 10, size=(3, 3))
print("A:\n", A)
print("B:\n", B)

# 2. Prodotto matriciale
C = A @ B
print("C = A @ B:\n", C)

# 3. Inversa di A (gestione errore se singolare)
try:
    A_inv = np.linalg.inv(A)
    print("Inversa di A:\n", A_inv)
except np.linalg.LinAlgError:
    print("A è singolare, non ha inversa.")

# 4. Autovalori di C
eigvals = np.linalg.eigvals(C)
print("Autovalori di C:", eigvals)
```

---

## 4. Dati Random e Analisi Statistica
**Obiettivo:**
1. Generare 1000 campioni da una distribuzione normale (media 0, std 1).
2. Calcolare media, varianza, deviazione standard.
3. Creare un istogramma (10 bin) usando `np.histogram`.
4. Verificare la percentuale di valori entro 1 σ.

```python
import numpy as np

# 1. Campioni normali
rng = np.random.default_rng()
samples = rng.normal(loc=0.0, scale=1.0, size=1000)

# 2. Statistiche di base
mean = samples.mean()
var = samples.var()
std = samples.std()
print(f"Media={mean:.3f}, Varianza={var:.3f}, Std={std:.3f}")

# 3. Istogramma
hist, bin_edges = np.histogram(samples, bins=10)
print("Istogramma (conteggi per bin):", hist)
print("Bordi dei bin:", bin_edges)

# 4. Percentuale entro 1 sigma
within_one_sigma = np.abs(samples - mean) <= std
percent = within_one_sigma.mean() * 100
print(f"Percentuale entro ±1σ: {percent:.1f}%")
```

---

## 5. Trasformata di Fourier (FFT) su un Segnale Senoidale
**Obiettivo:**
1. Creare un segnale composto da due sinusoidi: 5 Hz e 20 Hz, campionato a 200 Hz per 2 s.
2. Calcolare la FFT con `np.fft.rfft`.
3. Identificare le due frequenze dominanti dallo spettro.

```python
import numpy as np
import matplotlib.pyplot as plt

# Parametri
fs = 200          # Hz (frequenza di campionamento)
T = 2.0           # durata in secondi
t = np.arange(0, T, 1/fs)

# Segnale: 5 Hz + 20 Hz
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t)

# FFT
fft_vals = np.fft.rfft(signal)
fft_freq = np.fft.rfftfreq(len(signal), d=1/fs)

# Amplitudine normale
amp = np.abs(fft_vals) / len(signal)

# Trova i picchi più alti
peak_idxs = np.argsort(amp)[-2:]
print("Frequenze dominanti:", fft_freq[peak_idxs])
```

---

## 6. Confronto Performance: Loop vs Vectorizzazione
**Obiettivo:**
1. Sommare 10 000 000 di numeri usando un ciclo `for` tradizionale.
2. Ripetere la stessa operazione con `np.sum` su un array.
3. Misurare i tempi con `time.perf_counter`.

```python
import numpy as np
import time

N = 10_000_000
# Dati
data = np.arange(N, dtype=np.float64)

# 1. Loop tradizionale
start = time.perf_counter()
total_loop = 0.0
for x in data:
    total_loop += x
elapsed_loop = time.perf_counter() - start
print(f"Somma con loop: {total_loop:.0f} (tempo {elapsed_loop:.3f}s)")

# 2. Vectorizzazione
start = time.perf_counter()
total_vec = np.sum(data)
elapsed_vec = time.perf_counter() - start
print(f"Somma con np.sum: {total_vec:.0f} (tempo {elapsed_vec:.6f}s)")
```

---

*Questi esercizi sono pensati per far consolidare le basi di NumPy e introdurre concetti più avanzati di indicizzazione, operazioni lineari, statistica, trasformate di Fourier e ottimizzazione delle performance.*

---

## 7. Analisi di un dataset di vendite (NumPy + CSV)
**Scenario reale:** hai un CSV con dati di vendite giornaliere (data, prodotto, quantità, prezzo unitario). Vuoi calcolare:
1. **Fatturato totale per mese**.
2. **I 3 prodotti più venduti** in termini di fatturato.
3. **Media mobile a 7 giorni** del fatturato giornaliero.

```python
import numpy as np
import datetime as dt

# 1. Caricamento dati (date come stringhe)
# Supponi un file 'vendite.csv' con colonne: data,prodotto,qt,prezzo
# Esempio di dati simulati
np.random.seed(0)
num_giorni = 180
date_arr = np.arange(num_giorni, dtype='datetime64[D]') + np.datetime64('2023-01-01')
prodotti = np.random.choice(['A', 'B', 'C', 'D', 'E'], size=num_giorni)
qt = np.random.randint(1, 20, size=num_giorni)
prezzo = np.random.uniform(10, 100, size=num_giorni)

# 2. Calcolo fatturato per riga
fatturato = qt * prezzo

# 3. Fatturato per mese (groupby su month)
mesi = date_arr.astype('datetime64[M]')
unique_mesi = np.unique(mesi)
fatturato_mese = np.array([fatturato[mesi == m].sum() for m in unique_mesi])
print('Fatturato per mese:')
for m, f in zip(unique_mesi, fatturato_mese):
    print(m, f)

# 4. Top 3 prodotti per fatturato totale
# Convertiamo prodotti in array di bytes per indicizzazione veloce
prod_unique, prod_idx = np.unique(prodotti, return_inverse=True)
fatt_per_prodotto = np.zeros_like(prod_unique, dtype=float)
np.add.at(fatt_per_prodotto, prod_idx, fatturato)
top3_idx = np.argsort(fatt_per_prodotto)[-3:][::-1]
print('\nTop 3 prodotti per fatturato:')
for i in top3_idx:
    print(prod_unique[i], fatt_per_prodotto[i])

# 5. Media mobile a 7 giorni del fatturato giornaliero
window = 7
cumsum = np.cumsum(np.insert(fatturato, 0, 0))
media_mobile = (cumsum[window:] - cumsum[:-window]) / window
print('\nMedia mobile (7 giorni) primo 5 valori:', media_mobile[:5])
```

---

## 8. Istogramma di un'immagine in scala di grigi
**Scenario reale:** vuoi analizzare la distribuzione dei valori di intensità di un'immagine (es. per bilanciamento dei contrasti).

```python
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

# Carica immagine in scala di grigi (assicurati che l'immagine sia in 8‑bit)
img = imread('immagine_grigia.png')  # shape (H, W)

# Flatten e calcola l'istogramma (256 bins per valore 0‑255)
hist, bin_edges = np.histogram(img.ravel(), bins=256, range=(0, 255))

print('Istogramma (conteggi per livello di grigio):')
print(hist)

# Visualizza con matplotlib (opzionale)
plt.bar(bin_edges[:-1], hist, width=1, edgecolor='black')
plt.title('Istogramma intensità')
plt.xlabel('Valore pixel')
plt.ylabel('Frequenza')
plt.show()
```

---

## 9. Simulazione Monte Carlo per valutare il valore atteso di un portafoglio
**Scenario reale:** un portafoglio contiene due asset con rendimenti giornalieri normalmente distribuiti. Vuoi stimare il valore atteso e la varianza di un investimento di 10 000 € dopo 1 anno (252 giorni di trading).

```python
import numpy as np

np.random.seed(42)
N = 100_000  # simulazioni
T = 252      # giorni di trading
# Parametri asset
mu = np.array([0.0005, 0.0003])   # rendimento medio giornaliero
sigma = np.array([0.01, 0.015])   # volatilità giornaliera

# Correlazione tra asset
corr = 0.3
cov = np.diag(sigma**2)
cov[0, 1] = cov[1, 0] = corr * sigma[0] * sigma[1]

# Simulazione dei ritorni giornalieri (multivariato normale)
returns = np.random.multivariate_normal(mu, cov, size=(N, T))
# Valore cumulativo per asset
cum_returns = np.cumprod(1 + returns, axis=1)
# Investimento iniziale uguale per asset
initial = 5_000
portfolio = initial * cum_returns[:, :, 0] + initial * cum_returns[:, :, 1]
final_values = portfolio[:, -1]

print('Valore medio finale: €', final_values.mean())
print('Deviazione standard: €', final_values.std())
```

---

## 10. Rilevamento outlier in una serie temporale di temperature
**Scenario reale:** hai una lista di temperature giornaliere e vuoi identificare giorni con valori anomali (outlier) usando l’intervallo interquartile (IQR).

```python
import numpy as np

np.random.seed(1)
# Simulazione temperature con outlier
temp = np.random.normal(20, 5, size=365)
# Aggiungi outlier estremi
temp[[50, 200, 300]] = [5, 35, 40]

Q1 = np.percentile(temp, 25)
Q3 = np.percentile(temp, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = np.where((temp < lower) | (temp > upper))[0]

print('Outlier indices (giorni):', outliers)
print('Valori outlier:', temp[outliers])
```

---

*Questi nuovi esercizi mostrano come NumPy possa essere applicato a casi d’uso tipici di analisi dati, elaborazione immagini, simulazioni finanziarie e pulizia di serie temporali.*
