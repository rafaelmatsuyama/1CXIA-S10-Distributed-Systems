# Lab 03a: A Máquina Infinita (Provisionamento Elástico)

**Disciplina:** Sistemas Distribuídos para IA (1CXIA - S10)
**Contexto:** Cloud Computing & Azure Machine Learning
**Duração Estimada:** 60 minutos

---

## 🎯 Objetivo do Lab
Provar a elasticidade da nuvem através do provisionamento automatizado de recursos de computação que "nascem" sob demanda e "morrem" em períodos de ociosidade, otimizando o custo operacional.

## 🏦 Cenário CAIXA
A equipe de Crédito Imobiliário precisa treinar um modelo de propensão de vendas para o **Feirão da Casa Própria**. O volume de dados é massivo, mas o treinamento ocorre apenas uma vez por semana. Manter um servidor potente ligado 24/7 seria um desperdício de orçamento público. Precisamos de um cluster que escale de **0 a 4 nós** automaticamente.

---

## 🛠️ Passo a Passo

### 💻 Acesso ao Azure Cloud Shell
1. Acesse o **Azure Portal** ([portal.azure.com](https://portal.azure.com)).
2. Clique no ícone do **Cloud Shell** (`>_`) na barra superior, ao lado da barra de busca.
3. Se for o primeiro acesso, selecione **Bash** e siga as instruções para criar o armazenamento (Storage) necessário.
4. No canto superior esquerdo do terminal, garanta que esteja selecionado **Bash** (e não PowerShell).

### 1. Preparação do Terreno (Resource Group & Workspace)
No terminal do **Azure Cloud Shell**, vamos criar o ambiente base:

```bash
# 1. Definir variáveis (Ajuste o sufixo para ser único)
ID=$RANDOM
RG="rg-1cxia-caixa-$ID"
LOC="brazilsouth"
WS="ws-ml-caixa-$ID"

# 2. Criar o Resource Group (O "Terreno")
az group create --name $RG --location $LOC

# 3. Criar o Workspace de Machine Learning (A "Sede")
az ml workspace create --name $WS --resource-group $RG --location $LOC
```

### 2. Criando a "Máquina Infinita" (Compute Cluster)
Agora, vamos provisionar o cluster que realmente processa os dados. O segredo está no parâmetro `--min-instances 0`.

```bash
# Criar cluster de CPU que escala conforme a fila de jobs
az ml compute create --name "cluster-elastico" \
    --workspace-name $WS \
    --resource-group $RG \
    --type amlcompute \
    --size STANDARD_DS11_V2 \
    --min-instances 0 \
    --max-instances 4 \
    --tier Dedicated
```

### 3. O Teste de Estresse (Scale-up)
Para forçar a Azure a ligar as máquinas do seu cluster, vamos submeter um "Heavy Script" de 5 minutos.

1. No terminal do **Cloud Shell**, crie o script de processamento:
```bash
cat <<EOF > stress.py
import time
import math
import os

print(f"--- Iniciando Processamento Pesado no No: {os.uname().nodename} ---")
# Simula carga de CPU por 5 minutos
end_time = time.time() + 300
while time.time() < end_time:
    _ = [math.sqrt(i) for i in range(1000000)]
print("--- Processamento Concluido com Sucesso! ---")
EOF
```

2. Crie o arquivo de definição do Job (`job.yml`):
```bash
cat <<EOF > job.yml
command: python stress.py
code: .
environment: azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest
compute: azureml:cluster-elastico
display_name: teste-estresse-caixa
experiment_name: aula-03-escala
EOF
```

3. Submeta o Job para o seu Workspace:
```bash
az ml job create --file job.yml --workspace-name $WS --resource-group $RG
```

### 📈 Monitoramento em Tempo Real (Onde a Mágica Acontece)
Não basta submeter e esperar. Um Engenheiro de MLOps monitora o "pulso" da infraestrutura.

1. **Visão do Orquestrador (Compute Scale):**
   - No [Azure ML Studio](https://ml.azure.com), vá em **Compute > Compute Clusters**.
   - Observe o `cluster-elastico`: ele passará de **0** para **Resizing** e depois para **1 (ou mais) nós ativos**. 
   - Note que o estado do nó muda de *Idle* para *Running* assim que o Job é alocado.

2. **Visão do Job (Logs & Streaming):**
   - Vá na aba **Jobs** no menu lateral esquerdo.
   - Clique no Job chamado `teste-estresse-caixa`.
   - Clique na aba **Outputs + logs**.
   - Abra o arquivo `user_logs/std_log.txt`. Aqui você verá o script `stress.py` imprimindo o nome do nó e o progresso em tempo real.

3. **Visão do Hardware (Métricas):**
   - Na página do Job, clique em **Monitoring**.
   - Observe os gráficos de **CPU Utilization**. Você verá o salto para quase 100% de uso assim que o script pesado iniciar.

4. **O Scale-to-Zero:**
   - Após o Job terminar (Status: *Completed*), aguarde cerca de 2 minutos sem fazer nada.
   - Volte em **Compute Clusters** e veja o cluster voltando sozinho para **0 nós**.

---

## 🏆 Desafio Tier 2 (Otimização de Custo)
Modelos de treinamento não críticos podem rodar em hardware "sobrante" da Azure por uma fração do preço.
*   **Tarefa:** Crie um segundo cluster chamado `cluster-economico` usando a flag `--tier lowpriority`. 
*   **Discussão:** Qual o risco de usar *Low Priority* (Spot VMs) para um sistema de atendimento em tempo real da CAIXA?

---

## 🧠 Gabarito e Discussão
*   **O que aconteceu?** Quando o Job foi submetido, o orquestrador da Azure detectou a fila e disparou o provisionamento de VMs.
*   **Scale-to-Zero:** Após 120 segundos (padrão) de inatividade, o cluster volta para 0 nós.
*   **DNA Pedagógico:** A nuvem não é "computador dos outros", é **automação de hardware**. Se você não automatiza o desligamento, você está apenas usando um Data Center caro.

---

## 🧹 Limpeza (Obrigatório)
Para evitar cobranças na conta de estudante/corporativa:
```bash
az group delete --name $RG --yes --no-wait
```
