# Inżynieria danych / Business Intelligence

## 6. Metody integracji danych

Integracja danych to proces łączenia danych pochodzących z różnych źródeł w spójną i ujednoliconą formę, wykorzystywaną w systemach analitycznych, hurtowniach danych czy aplikacjach BI (Business Intelligence). Jest kluczowa w kontekście organizacji przetwarzających dane z wielu systemów (np. ERP, CRM, IoT, relacyjne bazy danych, API itp.).

---

### Główne podejścia do integracji danych

#### 1. ETL (Extract, Transform, Load)
Najczęściej stosowana metoda w hurtowniach danych:
- **Extract** – pobieranie danych ze źródeł (np. SQL, NoSQL, API, CSV).
- **Transform** – czyszczenie, mapowanie, ujednolicanie.
- **Load** – załadowanie do repozytorium (np. Snowflake, Redshift).

*Narzędzia ETL*: Apache NiFi, Talend, Microsoft SSIS, Informatica, dbt, Apache Airflow

#### 2. ELT (Extract, Load, Transform)
Stosowane w nowoczesnych rozwiązaniach, gdzie dane najpierw ładowane są w surowej postaci do magazynu (np. Data Lake), a dopiero potem transformowane. Umożliwia zachowanie pełnej historii danych.

#### 3. Replikacja danych
Kopiowanie danych między systemami z zastosowaniem mechanizmów takich jak **Change Data Capture (CDC)**.

*Narzędzia*: Debezium, Fivetran, StreamSets

#### 4. Federacja danych
Zapytania do danych w wielu źródłach bez ich przenoszenia, wykonywane w czasie rzeczywistym.

*Technologie*: Presto/Trino, Denodo, Apache Drill

#### 5. Wirtualizacja danych
Użytkownik uzyskuje ujednolicony widok na dane z różnych źródeł, bez fizycznej integracji.

#### 6. Integracja strumieniowa
Obsługa danych w czasie rzeczywistym przy użyciu streamingu.

*Technologie*: Apache Kafka, Kafka Connect, Flink, Spark Streaming

#### 7. Integracja aplikacyjna
Synchronizacja danych za pomocą API i middleware (np. Apache Camel, Mulesoft).

---

### Jeziora danych (Data Lakes)

**Data Lake** to centralne repozytorium umożliwiające przechowywanie danych w dowolnym formacie – zarówno surowych, jak i przetworzonych (strukturalnych, półstrukturalnych i niestrukturalnych). Cechy:
- Brak sztywnej struktury (schema-on-read),
- Obsługa dużych wolumenów danych (Big Data),
- Przechowywanie danych do późniejszej analizy lub uczenia maszynowego.

*Technologie*: AWS S3, Azure Data Lake Storage, Hadoop HDFS, Delta Lake, Apache Iceberg

Data Lake często współpracuje z **Data Lakehouse**, które łączy cechy Lake i Hurtowni Danych (schema-on-write + elastyczność Lake). Popularnym rozwiązaniem jest np. **Databricks Lakehouse**.

---

### Wybrane techniki i algorytmy
- **Mapowanie schematów** – przypisanie pól między różnymi źródłami danych.
- **Deduplikacja** – wykrywanie rekordów reprezentujących te same encje (alg. Levenshteina, Soundex).
- **Standaryzacja** – ujednolicenie formatów, jednostek, nazw.

---

### Zastosowania
- Budowa hurtowni danych (Data Warehouse)
- Zasilanie **Data Lakes** dla Big Data i AI
- Konsolidacja danych w BI (np. Tableau, Power BI)
- Migracja danych między systemami
- Tworzenie **pipelinów danych** do ML/AI

---

## 7. Porównanie: Jezioro danych (Data Lake) vs Hurtownia danych (Data Warehouse)

W systemach analitycznych i Business Intelligence dwa główne podejścia do przechowywania danych to **jeziora danych (data lakes)** oraz **hurtownie danych (data warehouses)**. Oba mają na celu centralizację i dostęp do danych, ale różnią się strukturą, zastosowaniem oraz technologiami.

---

### Hurtownia danych (Data Warehouse)

**Cechy:**
- **Struktura danych:** Dane są zorganizowane, przetworzone i sformatowane (schema-on-write).
- **Zastosowanie:** Głównie analizy biznesowe (BI), raportowanie, OLAP.
- **Wydajność:** Optymalizowana do zapytań analitycznych.
- **Integracja danych:** ETL (Extract, Transform, Load).

**Technologie:**
- **Relacyjne bazy danych:** PostgreSQL, Oracle, MS SQL Server, Amazon Redshift, Snowflake.
- **ETL Tools:** Apache NiFi, Talend, Informatica, dbt.
- **Warstwy przetwarzania:** Star Schema, Snowflake Schema, OLAP Cubes.

**Przykładowe algorytmy:**
- Algorytmy agregacji (np. ROLLUP, CUBE).
- Analiza trendów: regresja liniowa, ARIMA.
- Klasyczne raportowanie BI: zapytania SQL, eksploracja danych.

---

### Jezioro danych (Data Lake)

**Cechy:**
- **Struktura danych:** Składowanie danych w stanie surowym (schema-on-read), obsługa danych strukturalnych, półstrukturalnych (JSON, XML), niestrukturalnych (obrazy, logi).
- **Zastosowanie:** Big Data, Machine Learning, eksploracja danych, AI.
- **Elastyczność:** Możliwość pracy na danych surowych, wsparcie dla danych strumieniowych.
- **Integracja danych:** ELT (Extract, Load, Transform).

**Technologie:**
- **Magazyny danych:** Hadoop HDFS, Amazon S3, Azure Data Lake Storage.
- **Silniki przetwarzania:** Apache Spark, Apache Flink, Presto, Dask.
- **Platformy ML/AI:** TensorFlow, PyTorch, scikit-learn.

**Przykładowe algorytmy:**
- Uczenie nadzorowane: Random Forest, XGBoost, Support Vector Machines (SVM), sieci neuronowe (MLP, CNN).
- Uczenie nienadzorowane: KMeans, DBSCAN, PCA, Autoenkodery.
- Analiza strumieniowa: Apache Kafka + Flink/Spark Streaming.

---

### Kluczowe różnice

| Cecha                         | Hurtownia danych                  | Jezioro danych                        |
|------------------------------|----------------------------------|---------------------------------------|
| Struktura danych             | Strukturalna, ustandaryzowana     | Surowa, dowolna                       |
| Czas przetwarzania danych    | Przed załadunkiem (ETL)           | Przy odczycie (ELT)                   |
| Koszt przechowywania         | Wyższy                            | Niższy                                |
| Elastyczność i skalowalność  | Ograniczona                       | Wysoka                                |
| Obsługa danych nienumerycznych | Słaba                           | Bardzo dobra                          |
| Użycie                       | BI, raportowanie                  | AI, Big Data, Data Science            |

---

### Wnioski

- **Data Warehouse** to sprawdzona architektura do analiz i raportowania w środowiskach biznesowych, gdzie wymagane są czyste i znormalizowane dane.
- **Data Lake** jest elastyczną alternatywą, lepiej dopasowaną do projektów AI/ML oraz do integracji dużych wolumenów danych z wielu źródeł.

W nowoczesnych systemach często stosuje się **podejścia hybrydowe** (np. **Lakehouse** – Databricks, Delta Lake), łączące zalety obu modeli.


## 8. Twierdzenie CAP
