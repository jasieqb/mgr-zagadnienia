# Machine Learning (Uczenie Maszynowe)

## 1. Paradygmaty (rodzaje) uczenia maszynowego.


### Supervised Learning (uczenie nadzorowane)

- **Dane**: oznakowane pary (wejście → etykieta/wyjście).
- **Typowe zadania i algorytmy**:
  - **Klasyfikacja**: Logistic Regression, SVM, Decision Tree, Random Forest, k‑Nearest Neighbors, Naive Bayes, Convolutional Neural Networks (CNN).
  - **Regresja**: Linear Regression, Ridge/Lasso, Gaussian Process Regression.
- **Metody ensemble**: Bagging (Random Forest), Boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM).
- **Zastosowania**: filtrowanie spamu, wycena nieruchomości, wykrywanie chorób, rozpoznawanie obrazów.

---

### Unsupervised Learning (uczenie nienadzorowane)

- **Dane**: nieoznakowane.
- **Algorytmy i techniki**:
  - **Klasteryzacja**: k‑means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Models.
  - **Redukcja wymiarów**: PCA, t‑SNE, ICA, LDA, autoenkodery.
  - **Wykrywanie anomalii**: Isolation Forest, Local Outlier Factor.
  - **Asocjacja**: Apriori, FP‑growth (analiza koszykowa).
- **Zastosowania**: segmentacja klientów, wizualizacja danych, wykrywanie fraudów.

---

### Reinforcement Learning (uczenie przez wzmacnianie)

- **Mechanizm**: agent działa w środowisku, otrzymuje nagrody/kary i optymalizuje strategię.
- **Algorytmy klasyczne**:
  - *Model‑free*: Q‑learning, SARSA.
  - *Model‑based*: Dyna, MuZero.
- **Deep RL**: Deep Q-Network (DQN), Double DQN, Policy Gradient, Actor–Critic (A2C, PPO), DDPG, SAC.
- **Zastosowania**: gry (Atari, AlphaGo), robotyka, autonomiczne pojazdy, finanse, RLHF (jak w ChatGPT).

---

### Paradygmaty hybrydowe i rozszerzenia

- **Semi‑supervised**: self-training, co-training, metody grafowe.
- **Self‑supervised**: maskowanie tokenów (BERT), rotacje obrazów (SimCLR, BYOL).
- **Meta‑learning**, **Online learning**, **Weak supervision**, **Genetic programming**.

---

### 📊 Tabela porównawcza

| Paradygmat         | Dane                  | Przykładowe algorytmy                                               | Typowe zastosowania                     |
|--------------------|-----------------------|-----------------------------------------------------------------------|------------------------------------------|
| **Supervised**     | oznakowane            | Logistic Regression, SVM, Random Forest, CNN                         | spam, wycena, diagnostyka, klasyfikacja |
| **Unsupervised**   | nieoznakowane         | k‑means, PCA, autoenkodery, Isolation Forest                        | segmentacja, wizualizacja, wykrywanie anomalii |
| **Reinforcement**  | interakcja + nagroda  | Q‑learning, DQN, PPO, MuZero                                        | gry, robotyka, sterowanie, finanse      |
| **Hybrydowe**      | różne                 | Semi‑sup, self‑sup, meta‑learning, weak supervision                  | NLP, medycyna, transfer learning         |

---

### ✅ Wnioski

- **Supervised**: gdy masz oznakowane dane i chcesz nauczyć model konkretnego zadania predykcyjnego.
- **Unsupervised**: gdy dane są nieoznaczone i chcesz odkryć strukturę bez kierowania.
- **Reinforcement**: gdy model może uczyć się poprzez interakcję i optymalizację nagród.
- **Hybrydowe**: efektywne, gdy etykiet jest mało lub zależy Ci na reprezentacji.

---

## 2. Pojęcie parametrów i hiperparametrów modelu uczenia maszynowego – modele parametryczne i nieparametryczne.


### 🔍 Parametry w modelach parametrycznych i nieparametrycznych

#### Modele parametryczne

W modelach **parametrycznych** zakładamy z góry określoną formę modelu (np. funkcja liniowa, sigmoidalna) i uczymy jedynie **skończoną liczbę parametrów**, niezależnie od rozmiaru danych.

- **Liczba parametrów jest stała**
- Parametry są bezpośrednio dostrajane w procesie treningu
- Model może być **szybki w trenowaniu**, ale ograniczony w elastyczności (np. nie odwzoruje skomplikowanych zależności)

🔧 **Przykłady parametrów:**
- Regresja liniowa: współczynniki $w_1, w_2, \ldots, w_n$, bias $b$
- Regresja logistyczna: te same co wyżej, ale dla klasyfikacji
- Sieci neuronowe (MLP): macierze wag i biasy

#### Modele nieparametryczne

W modelach **nieparametrycznych** nie zakładamy konkretnej formy funkcji, a liczba „parametrów” **rośnie wraz z ilością danych**. Takie modele „zapamiętują” dane i opierają decyzje na ich analizie w czasie rzeczywistym.

- **Brak stałej liczby parametrów**
- Model nie uczy się konkretnych wag — decyzje opiera na strukturze danych
- Modele często są **bardziej elastyczne**, ale **wolniejsze przy predykcji**

🔍 **Przykłady zachowania parametrów:**
- KNN: nie ma parametrów trenowanych; do klasyfikacji wykorzystuje całe dane treningowe
- Drzewa decyzyjne: struktura drzewa (głębokie drzewo = dużo gałęzi) zależy od danych
- SVM z RBF: parametry granicy decyzyjnej zależą od punktów wspierających (support vectors), które są wybierane z danych

---

### 🔄 Porównanie

| Cecha                    | Model parametryczny         | Model nieparametryczny        |
|--------------------------|-----------------------------|-------------------------------|
| Liczba parametrów        | Stała                       | Zmienna (zależna od danych)   |
| Uczenie                  | Bezpośrednia optymalizacja | "Zapamiętywanie" danych       |
| Elastyczność             | Ograniczona                 | Wysoka                        |
| Prędkość predykcji       | Szybka                      | Wolniejsza                    |
| Przykłady                | Regresja, MLP, SVM (lin.)   | KNN, Drzewa, SVM (RBF)        |

---

### 📌 Wnioski

- W **modelach parametrycznych** parametry są *głównym nośnikiem wiedzy*, a ich liczba jest ustalona — dobór odpowiedniej architektury jest kluczowy.
- W **modelach nieparametrycznych** "parametry" są dynamiczne, a wiedza jest *rozproszona w danych* — modele te lepiej radzą sobie z bardziej złożonymi rozkładami, kosztem wydajności.

---

## 3. Regresja i klasyfikacja – algorytmy uczenia nadzorowanego

Uczenie nadzorowane (*supervised learning*) to podejście w uczeniu maszynowym, w którym model uczy się odwzorowywać relację pomiędzy danymi wejściowymi a etykietami (wynikami), na podstawie zbioru treningowego zawierającego pary (X, y). Wyróżniamy dwa główne typy problemów:

- **Regresja** – gdy etykieta (y) jest zmienną ciągłą.
- **Klasyfikacja** – gdy etykieta jest dyskretna (np. 0/1, klasa A/B/C).

---

## 🟩 Regresja

**Regresja** polega na przewidywaniu wartości liczbowej. Przykłady:
- prognoza cen (np. mieszkań),
- przewidywanie temperatury, zapotrzebowania, zysków itp.

### Przykładowe algorytmy regresji:
- **Regresja liniowa** – dopasowuje prostą linię do danych minimalizując błąd średniokwadratowy.
- **Ridge Regression** – regresja liniowa z karą L2, ogranicza rozmiar współczynników.
- **Lasso Regression** – regresja z karą L1, może zerować nieistotne cechy.
- **Decision Tree Regressor** – tworzy reguły dzielące dane i przypisuje wartość w liściach.
- **Random Forest Regressor** – agreguje wiele drzew regresyjnych dla większej stabilności.
- **Support Vector Regression (SVR)** – dopasowuje model liniowy z tolerancją błędu (marginesem).
- **Sieci neuronowe (np. MLPRegressor)** – modelują nieliniowe zależności między zmiennymi.
- **XGBoost / LightGBM / CatBoost** – nowoczesne, wydajne algorytmy boostingowe o wysokiej skuteczności.

---

## 🟦 Klasyfikacja

**Klasyfikacja** polega na przypisaniu danych wejściowych do jednej z klas. Przykłady:
- rozpoznawanie cyfr na obrazach,
- diagnoza chorób,
- wykrywanie spamu, klasyfikacja dokumentów.

### Przykładowe algorytmy klasyfikacji:
- **Logistic Regression** – estymuje prawdopodobieństwo przynależności do klasy.
- **Decision Tree Classifier** – buduje drzewo decyzji na podstawie reguł podziału.
- **Random Forest Classifier** – tworzy wiele drzew i głosuje większością.
- **Support Vector Machine (SVM)** – znajduje granicę maksymalizującą margines między klasami.
- **K-Nearest Neighbors (KNN)** – klasyfikuje na podstawie najbliższych sąsiadów.
- **Naive Bayes** – prosty klasyfikator probabilistyczny oparty na twierdzeniu Bayesa.
- **Sieci neuronowe (np. MLPClassifier, CNN)** – uczą się nieliniowych relacji, skuteczne zwłaszcza przy dużych i złożonych danych (np. obrazy, dźwięk).
- **XGBoost / LightGBM / CatBoost** – bardzo wydajne algorytmy boostingowe, często osiągające najlepsze wyniki w zadaniach klasyfikacyjnych.

---

## 🔧 Technologie i biblioteki

Do implementacji i eksperymentów najczęściej wykorzystywane są:
- **Python + Scikit-learn** – klasyczne algorytmy ML.
- **XGBoost / LightGBM / CatBoost** – szybkie i skuteczne boostery.
- **TensorFlow / PyTorch / Keras** – głębokie sieci neuronowe.
- **H2O.ai, WEKA** – narzędzia GUI lub JVM do eksploracji danych.
- **AutoML (np. AutoSklearn, H2O AutoML, Google AutoML)** – automatyczny dobór modeli i hiperparametrów.


---

## 4. Algorytmy uczenia nienadzorowanego

## Algorytmy uczenia nienadzorowanego

Uczenie nienadzorowane (ang. *unsupervised learning*) to paradygmat uczenia maszynowego, w którym model uczy się struktur i wzorców z danych **bez etykiet**. Celem jest odkrycie ukrytych zależności, struktur, klastrów lub redukcja wymiarowości danych.

Typowe zastosowania:

- Klasteryzacja (grupowanie danych)
- Redukcja wymiarowości (ekstrakcja cech)
- Detekcja anomalii
- Reguły asocjacyjne (analiza koszykowa)

---

### Kluczowe algorytmy i metody

#### 📊 Klasteryzacja

1. **K-means** – dzieli dane na *k* klastrów, minimalizując wewnątrzgrupową wariancję.
2. **DBSCAN** – klasteryzacja oparta na gęstości; wykrywa klastry dowolnych kształtów.
3. **Mean Shift** – przesuwanie punktów do centrum gęstości.
4. **Agglomerative Clustering** – metoda hierarchiczna; łączy punkty w klastry.
5. **Spectral Clustering** – używa wartości własnych macierzy podobieństwa.

#### 📉 Redukcja wymiarowości

1. **PCA** – transformacja danych do ortogonalnych komponentów.
2. **t-SNE** – wizualizacja w 2D/3D, zachowująca lokalną strukturę.
3. **UMAP** – szybka, nieliniowa metoda do wizualizacji i klasteryzacji.
4. **Autoenkodery** – sieci neuronowe do kompresji danych.

---

### Autoenkodery (Autoencoders)

**Autoenkodery** to rodzaj sztucznych sieci neuronowych służących do **uczenia reprezentacji danych** w sposób nienadzorowany. Ich celem jest nauczenie się zakodowanej, skompresowanej reprezentacji wejścia poprzez jego rekonstrukcję.

#### Struktura autoenkodera

Autoenkoder składa się z trzech głównych komponentów:

1. **Encoder** – przekształca dane wejściowe w zakodowaną reprezentację (wektor cech).
2. **Latent space (kod, wektor ukryty)** – niskowymiarowa reprezentacja danych (np. 16 lub 32 wartości zamiast 784 pikseli).
3. **Decoder** – odtwarza dane wejściowe na podstawie zakodowanej reprezentacji.

Schemat:
Wejście → Encoder → Latent space → Decoder → Wyjście (rekonstrukcja)

---

#### Zastosowania

- **Redukcja wymiarowości** (alternatywa dla PCA, ale nieliniowa)
- **Denoising Autoencoders** – usuwanie szumu z danych
- **Anomaly Detection** – analiza odchyleń między wejściem a rekonstrukcją
- **Generowanie danych** (np. obrazy, tekst)
- **Pre-trening w uczeniu głębokim** – wstępna kompresja/ekstrakcja cech

---

#### Typy autoenkoderów

1. **Vanilla Autoencoder** – najprostszy typ z gęstymi warstwami (Dense).
2. **Denoising Autoencoder** – uczy się rekonstrukcji danych, które wcześniej celowo zaszumiono.
3. **Sparse Autoencoder** – dodaje regularizację L1/L2, aby wymusić rzadką reprezentację.
4. **Variational Autoencoder (VAE)** – probabilistyczny model, koduje dane jako rozkłady, używany w generatywnych modelach (np. GANy).
5. **Convolutional Autoencoder** – oparty na warstwach konwolucyjnych; lepszy dla danych obrazowych.

---

#### 🧾 Reguły asocjacyjne (analiza koszykowa)

1. **Apriori** – znajduje częste zestawy przedmiotów i tworzy reguły asocjacyjne.
2. **FP-Growth** – efektywna alternatywa dla Apriori, działa bez generowania kandydatów.
3. **Eclat** – szybki algorytm oparty na przecięciu zbiorów transakcji.

Typowe metryki:

- **Support** – częstość występowania zbioru.
- **Confidence** – prawdopodobieństwo wystąpienia B przy A.
- **Lift** – siła zależności między A a B.

Zastosowania:

- E-commerce (polecanie produktów)
- Analiza zachowań zakupowych
- Optymalizacja układu sklepu

#### 🔍 Detekcja anomalii

1. **Isolation Forest** – losowe dzielenie danych w celu izolacji nietypowych punktów.
2. **One-Class SVM** – model granicy dla „normalnych” obserwacji.
3. **LOF (Local Outlier Factor)** – lokalna gęstość punktów vs. sąsiedzi.

---

### Technologie i biblioteki

- `scikit-learn` – PCA, klasteryzacja, LOF, IsolationForest
- `mlxtend` – implementacja algorytmu Apriori i reguł asocjacyjnych
- `PyCaret` – szybka eksploracja modeli ML, w tym unsupervised
- `Orange3` – GUI do eksploracji danych, w tym klasteryzacji i asocjacji
- `TensorFlow` / `PyTorch` – autoenkodery i inne modele nienadzorowane

---

## 5. Problemy przeuczenia i niedouczenia modelu uczenia maszynowego oraz metody zapobiegania

W procesie trenowania modeli ML kluczowym wyzwaniem jest znalezienie balansu między **niedouczeniem (underfitting)** a **przeuczeniem (overfitting)**. Oba te zjawiska można często zaobserwować poprzez analizę wykresów błędów: `loss` oraz `val_loss`.

---

### 🔍 Niedouczenie (Underfitting)

Model jest **zbyt prosty**, by nauczyć się struktury danych.

**Objawy:**
- Wysoki `loss` i `val_loss`, oba zmniejszają się powoli lub wcale.
- Niskie metryki (np. accuracy, F1) zarówno na danych treningowych, jak i walidacyjnych.

**Przyczyny:**
- Zbyt prosta architektura,
- Za mało epok trenowania,
- Zbyt silna regularizacja.

**Przykłady algorytmów:**
- Regresja liniowa, naiwny Bayes, zbyt płytkie drzewa.

---

### 🔍 Przeuczenie (Overfitting)

Model **dopasowuje się nadmiernie** do danych treningowych, ucząc się także szumu.

**Objawy:**
- `loss` na zbiorze treningowym maleje, ale `val_loss` rośnie po pewnym czasie.
- Model osiąga wysokie accuracy na treningu, ale niskie na walidacji.

**Typowy wykres:**

```
epoch
│ val_loss
│ /‾‾‾‾‾‾
│ /
│ /
│‾‾‾/ loss
│
└────────────────────
```

**Przykłady algorytmów narażonych:**
- Drzewa bez przycinania, sieci neuronowe bez `dropout`, KNN przy małym `k`.

---

### 🛠️ Metody zapobiegania

| Metoda | Opis | Przykład |
|--------|------|----------|
| **Early stopping** | Przerywanie trenowania, gdy `val_loss` przestaje się poprawiać | `EarlyStopping(patience=5)` w Keras |
| **Regularizacja** | Ograniczenie złożoności modelu | L1/L2 (`alpha` w Ridge/Lasso), `dropout` |
| **Cross-validation** | Ocena modelu na wielu podziałach danych | `cross_val_score()` |
| **Data augmentation** | Rozszerzenie zbioru treningowego | Transformacje danych wejściowych |
| **Redukcja złożoności** | Mniejsza liczba warstw, ograniczenie `max_depth` | W drzewach, sieciach |
| **Ensembles** | Agregacja prostszych modeli | Random Forest, Bagging, Boosting |

---

### 📈 Monitorowanie `loss`, `val_loss`, `val_accuracy`

W praktyce trenowania modeli (szczególnie z użyciem **Keras**, **PyTorch**) należy zawsze monitorować:
- `loss` – błąd na danych treningowych,
- `val_loss` – błąd na zbiorze walidacyjnym,
- `val_accuracy` lub inne metryki (np. F1, recall).

**Wskazówka inżynierska:** jeśli `val_loss` zaczyna rosnąć, a `loss` nadal maleje → zastosuj `early stopping`, `dropout`, lub zmniejsz model.

---

### 🧰 Technologie i biblioteki

- `scikit-learn` – klasyczne algorytmy i walidacja,
- `Keras` – `EarlyStopping`, `Dropout`, `ModelCheckpoint`,
- `XGBoost`, `LightGBM` – wbudowana walidacja, `early_stopping_rounds`,
- `TensorBoard` – wizualizacja `loss/val_loss` i metryk.

---
