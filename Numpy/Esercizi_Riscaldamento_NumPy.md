# Esercizi di Riscaldamento NumPy (Basi)

Questo file contiene una serie di esercizi semplici per prendere confidenza con NumPy (Slicing, Maschere Booleane, Modifica Elementi, Diagonali e Riduzioni).

---

## 1. Riscaldamento con lo Slicing (1D)
**Cosa fare:**
1. Crea un array contenente i numeri interi da `10` a `19` compresi.
2. Estrai solo gli elementi in posizione pari (indice 0, 2, 4...) usando lo slicing.
3. Inverti l'array originale.

```python
import numpy as np

# 1. Creazione
a = np.arange(10, 20)
print("Array originale:           ", a)

# 2. Elementi in posizione pari (passo 2)
pari = a[::2]
print("Elementi in posizione pari:", pari)

# 3. Inversione (passo negativo)
inverso = a[::-1]
print("Array invertito:           ", inverso)
```

---

## 2. Maschera Booleana Semplice (Filtro)
**Cosa fare:**
1. Crea un array con le temperature di una settimana: `[22, 18, 31, 29, 15, 27, 24]`.
2. Genera una maschera booleana che indichi quali giorni hanno avuto una temperatura **superiore a 25 gradi**.
3. Usa la maschera per estrarre solo quelle temperature calde.

```python
import numpy as np

# 1. Array temperature
temp = np.array([22, 18, 31, 29, 15, 27, 24])

# 2. Creazione della maschera booleana
caldo_mask = temp > 25
print("Maschera booleana (> 25°C):", caldo_mask)

# 3. Applicazione della maschera per filtrare i dati
temp_calde = temp[caldo_mask]
print("Temperature calde estratte:", temp_calde)
```

---

## 3. Modificare le Matrici (Slicing 2D)
**Cosa fare:**
1. Crea una matrice 4x4 composta interamente da zeri.
2. Modifica la matrice in modo che il quadrato interno (dimensioni 2x2) contenga solo il numero `9`.

```python
import numpy as np

# 1. Matrice 4x4 di zeri
m = np.zeros((4, 4), dtype=int)
print("Matrice iniziale:\n", m, "\n")

# 2. Modifica del quadrato interno usando lo slicing [righe, colonne]
m[1:3, 1:3] = 9
print("Matrice modificata:\n", m)
```

---

## 4. Diagonali Semplici
**Cosa fare:**
1. Crea una matrice 3x3 contenente i numeri da `1` a `9`.
2. Estrai i numeri presenti sulla diagonale principale.
3. Crea una nuova matrice diagonale 3x3 che abbia sulla diagonale i numeri `[100, 200, 300]`.

```python
import numpy as np

# 1. Creazione matrice 3x3
m = np.arange(1, 10).reshape(3, 3)
print("Matrice 3x3:\n", m, "\n")

# 2. Estrazione diagonale principale
diagonale = np.diag(m)
print("Diagonale estratta:", diagonale, "\n")

# 3. Creazione matrice diagonale a partire da un vettore 1D
nuova_diag = np.diag([100, 200, 300])
print("Nuova matrice diagonale:\n", nuova_diag)
```

---

## 5. Broadcasting per Conversioni
**Cosa fare:**
1. Crea un array di temperature in Celsius: `[0, 10, 20, 30, 40]`.
2. Converti tutte le temperature in Fahrenheit usando la formula: F = (C * 9/5) + 32.

```python
import numpy as np

# 1. Array Celsius
celsius = np.array([0, 10, 20, 30, 40])

# 2. Conversione automatica tramite Broadcasting (senza cicli for!)
fahrenheit = (celsius * 9/5) + 32
print("Temperature in Celsius:   ", celsius)
print("Temperature in Fahrenheit:", fahrenheit)
```

---

## 6. Medie Voti (Riduzione per Asse)
**Cosa fare:**
Abbiamo una matrice 3x4 che rappresenta i voti di **3 studenti** in **4 materie**:
```python
voti = np.array([[28, 24, 30, 26],  # Studente A
                 [30, 30, 28, 29],  # Studente B
                 [18, 20, 22, 18]]) # Studente C
```
1. Calcola la **media dei voti di ogni studente** (dovrai ottenere 3 medie).
2. Calcola la **media dei voti per ogni materia** (dovrai ottenere 4 medie).

```python
import numpy as np

voti = np.array([[28, 24, 30, 26],
                 [30, 30, 28, 29],
                 [18, 20, 22, 18]])

# 1. Media di ogni studente (riga). Riduce orizzontalmente le colonne (axis=1)
media_studenti = voti.mean(axis=1)
print("Media Studente A, B, C:", media_studenti)

# 2. Media di ogni materia (colonna). Riduce verticalmente le righe (axis=0)
media_materie = voti.mean(axis=0)
print("Media delle 4 Materie: ", media_materie)
```
