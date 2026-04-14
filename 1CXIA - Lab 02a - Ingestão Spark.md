# Lab 02a: O Motor do Big Data (Ingestão & ETL com PySpark)

**Disciplina:** Sistemas Distribuídos para IA (1CXIA - S10)
**Ambiente:** Databricks Free Edition
**Objetivo:** Configurar um cluster Spark, ingerir dados sintéticos e realizar operações de limpeza distribuída.

---

## 🛠️ Setup: Cadastro no Databricks Free Edition
Siga os passos abaixo para garantir seu ambiente de laboratório:
1.  Acesse: [databricks.com/try-databricks](https://www.databricks.com/try-databricks).
2.  Selecione **Get Free Edition**.
3.  Preencha seu e-mail pessoal.
4.  Na tela de escolha de **For Work** ou Cloud (AWS/Azure/GCP), **NÃO** selecione nenhuma. Clique no link: **"Get Free Edition"** (que redireciona para a modalidade **Free**).
5.  **Atenção:** O cluster desliga após 2h de inatividade. Salve seu trabalho!

---

## 🚀 Passo 1: Setup do Cluster
1.  No menu lateral, clique em **Compute**.
2.  Clique em **Play** no **Serverless Starter Warehouse**.
3.  Aguarde o ícone ficar verde (em alguns segundos).

## 📥 Passo 2: Ingestão de Dados (Cenário CAIXA)
Neste laboratório, simularemos a ingestão de transações bancárias.

1.  Crie um novo **Notebook** (Menu lateral -> + New -> Notebook).
2.  No primeiro bloco, vamos gerar dados sintéticos diretamente no Spark:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, rand, when

# O SparkSession já está disponível no Databricks como 'spark'

# Gerando 1 milhão de transações fictícias
df_transacoes = spark.range(1000000).withColumn("valor", (rand() * 1000).cast("decimal(10,2)")) \
    .withColumn("tipo", when(rand() > 0.5, lit("PIX")).otherwise(lit("TED"))) \
    .withColumn("status", when(rand() > 0.1, lit("CONCLUIDO")).otherwise(lit("FRAUDE")))

# Visualizando os dados de forma distribuÃ­da
display(df_transacoes)

# IMPORTANTE: Registrando o DataFrame para ser usado em SQL
df_transacoes.createOrReplaceTempView("df_transacoes")
```

## 🧹 Passo 3: ETL Distribuído (Transformações)
Lembre-se: O Spark utiliza **Lazy Evaluation**. Nada acontece até você chamar uma **Ação**.

1.  **Filtragem:** Selecione apenas transações de PIX com status CONCLUIDO.
2.  **Agregação:** Calcule a soma total de valores por tipo de transação.

```python
# Transformação 01: Filtro
df_pix_ok = df_transacoes.filter((col("tipo") == "PIX") & (col("status") == "CONCLUIDO"))

# Transformação 02: Agregação
df_resumo = df_transacoes.groupBy("tipo").sum("valor")

# AÇÃO: O Spark agora executa o plano de execução (DAG)
df_resumo.show()
```

## 📊 Passo 4: Visualização e SQL
O Databricks permite alternar entre Python e SQL de forma transparente.

```sql
%sql
-- Registrando o DataFrame como uma tabela temporária
CREATE OR REPLACE TEMPORARY VIEW v_transacoes AS SELECT * FROM df_transacoes;

-- Consulta SQL pura
SELECT tipo, COUNT(*) as total 
FROM v_transacoes 
GROUP BY tipo
```

---

## 🏆 Desafios (Tier 2)
1.  **Imputation:** Adicione uma nova coluna `tarifa` que seja 0.1% do `valor`, mas se o status for `FRAUDE`, a tarifa deve ser `null`. Em seguida, use `fillna()` para preencher os nulos com `0.0`.
    *   *💡 Dica:* Use a mesma lógica do `when().otherwise()` que utilizamos na criação do DataFrame inicial.
2.  **Performance:** Utilize o comando `df_transacoes.explain()` e tente identificar no Spark UI quantas tarefas (**Tasks**) foram geradas para o processamento de 1 milhão de linhas.
    *   *💡 Dica:* O Spark divide o trabalho em **Partitions**. O número de partições padrão no Spark SQL para shuffles é 200, mas para leitura simples, ele depende dos núcleos do cluster.

> [!IMPORTANT]
> **PARE!** Tente resolver os desafios acima por conta própria antes de prosseguir. A prática é o que consolida o aprendizado em sistemas distribuídos.

---

## ✅ Gabarito Comentado (Desafios)

### Solução Desafio 1: Imputation & Condicionais
Neste desafio, treinamos a manipulação de colunas baseada em condições lógicas e o tratamento de dados ausentes (essencial para MLOps).

```python
from pyspark.sql.functions import col, when, lit

# 1. Criando a coluna com condicional
df_desafio = df_transacoes.withColumn("tarifa", 
    when(col("status") == "FRAUDE", lit(None))
    .otherwise(col("valor") * 0.001)
)

# 2. Preenchendo os nulos (Imputation)
df_final = df_desafio.fillna(0.0, subset=["tarifa"])

display(df_final.filter(col("status") == "FRAUDE").select("valor", "status", "tarifa"))
```

### Solução Desafio 2: Plano de Execução (Explain)
O comando `explain()` é a ferramenta número 1 para o Engenheiro de Dados. Ele revela o plano físico que o Spark construiu.

```python
df_transacoes.explain(True)
```
*   **O que observar:** Procure por `Exchange` (indica Shuffling/Rede) e `FileScan` ou `Range` (indica a leitura dos dados). No Databricks Free, o número de partições costuma ser otimizado para os poucos núcleos disponíveis, minimizando o overhead de comunicação que vimos na Aula 01.

---

## 💡 Dica de Ouro
O comando `display(df)` no Databricks não é apenas um `print()`. Ele renderiza gráficos interativos. Tente clicar no ícone de gráfico (+) após rodar o `display` no Passo 2!
