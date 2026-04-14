# Lab 02b: IA em Escala (Modelo Híbrido: Spark + Scikit-Learn)

**Disciplina:** Sistemas Distribuídos para IA (1CXIA - S10)
**Ambiente:** Databricks Shared/Serverless
**Objetivo:** Processar volume massivo via Spark e treinar um modelo de fraude via Scikit-Learn no Driver.

---

## 🛠️ Por que este modelo Híbrido?
Em clusters **Shared/Serverless** (como a versão Free), o Spark MLlib possui restrições de segurança que bloqueiam seus construtores Java. Nossa estratégia de engenharia será:
1. **Spark:** Heavy lifting de dados (1 Milhão de linhas - Escala Horizontal).
2. **Pandas/Scikit-Learn:** Treinamento no Driver utilizando uma amostra segura para a memória (Escala Vertical).
3. **Pandas UDF:** Predição distribuída (O Spark enviando o modelo Python para os Workers).

---

## 🚀 Passo 1: Ingestão e ETL Massivo (Spark)
Geraremos 1 milhão de transações e realizaremos o pré-processamento distribuído no cluster.

```python
from pyspark.sql.functions import col, when, rand, lit

# 1. Gerando dados sintéticos (1 milhão de registros)
df_spark = spark.range(1000000).withColumn("valor", (rand() * 1000)) \
    .withColumn("tipo_index", when(rand() > 0.5, 1.0).otherwise(0.0)) \
    .withColumn("label", when(rand() > 0.9, 1.0).otherwise(0.0))

# 2. ETL Distribuído
df_clean = df_spark.filter(col("valor") > 0).select("valor", "tipo_index", "label")
print(f"Volume total no Spark: {df_clean.count()}")
```

## 📥 Passo 2: Conversão Segura (gRPC Limit)
No Databricks Serverless, o limite de tráfego entre o cluster e o Driver é de **128MB**. Para evitar o erro `RESOURCE_EXHAUSTED`, limitaremos a amostra para o treinamento.

```python
# Trazendo 100.000 linhas para o Driver (~60MB)
# 100k é uma amostra estatisticamente robusta para este modelo
pdf = df_clean.limit(100000).toPandas() 

print(f"Dados carregados no Driver: {len(pdf)} linhas.")
pdf.head()
```

## 🤖 Passo 3: Treinamento Local (Scikit-Learn)
Com o dado na memória do Driver, treinamos um detector de fraude clássico.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Divisão de dados
X = pdf[["valor", "tipo_index"]]
y = pdf["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento (No Driver - Pode levar entre 1 a 3 minutos)
model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model.fit(X_train, y_train)

print("Modelo treinado no Driver com sucesso!")
```

## 📊 Passo 4: Avaliação
```python
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"AUC-ROC: {auc:.4f}")
print(classification_report(y_test, y_pred))
```

---

## 🏆 Desafio (Tier 2): Predição Distribuída (Pandas UDF)
Como aplicar este modelo do Scikit-Learn em milhões de linhas sem converter tudo para Pandas?

**Dica:** Utilize as **Pandas UDFs**. Elas permitem que o Spark serialize seu modelo Python e o envie para todos os Workers processarem os dados em paralelo.

> [!IMPORTANT]
> **PARE!** Tente pesquisar sobre Pandas UDFs antes de ver o gabarito abaixo.

---

## ✅ Gabarito Comentado (Desafio)

### Solução: Pandas UDF
Nesta solução, o Spark gerencia o "tráfego" do modelo. Para evitar estourar o gRPC na visualização final, usamos um `limit` agressivo no `display`.

```python
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

# 1. Definindo a função de predição vetorizada
@pandas_udf(DoubleType())
def predict_fraud_udf(valor: pd.Series, tipo_index: pd.Series) -> pd.Series:
    # O Spark envia lotes (batches) de dados como séries do Pandas para os Workers
    X_batch = pd.DataFrame({"valor": valor, "tipo_index": tipo_index})
    return pd.Series(model.predict(X_batch))

# 2. Aplicando a predição em escala massiva (Rodando nos Workers)
# Visualizamos apenas 1.000 registros para garantir estabilidade do gRPC
df_final = df_spark.limit(1000).withColumn("predicao", 
                                           predict_fraud_udf(col("valor"), col("tipo_index")))

display(df_final.select("valor", "tipo_index", "predicao"))
```

---

## 💡 Resumo de Engenharia
* **ETL Massivo:** Spark (Escala Horizontal).
* **Treinamento:** Scikit-Learn (Amostra segura no Driver).
* **Predição:** Pandas UDF (Spark distribuindo o modelo Scikit-Learn).
* **Limitação Física:** O canal gRPC do Databricks Serverless é limitado a **128MB**. Respeite esse limite ao trazer dados para o Driver!
