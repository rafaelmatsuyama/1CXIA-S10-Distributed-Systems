# Lab 05b: O Segredo Compartilhado (Federated Learning)

**Disciplina:** Sistemas Distribuídos para IA (1CXIA - S10)
**Objetivo:** Simular o treinamento de um modelo de IA distribuído em 3 agências bancárias sem compartilhar dados sensíveis, utilizando o algoritmo Federated Averaging (FedAvg).
**Ambiente:** Google Colab (CPU/GPU) ou Databricks.

---

## 🚀 Instruções de Acesso ao Google Colab

Este laboratório pode ser realizado apenas com CPU, mas o uso de GPU acelera o treinamento local.

1.  Acesse: [https://colab.research.google.com/](https://colab.research.google.com/) e crie um **Novo Notebook**.
2.  (Opcional) **ATIVAR GPU:** Vá em `Ambiente de Execução` -> `Alterar o tipo de ambiente de execução` -> Selecione **T4 GPU** e clique em **Salvar**.

---

## 1. O Problema: Privacidade vs. Inteligência (LGPD)
A CAIXA precisa treinar um modelo de detecção de fraude altamente preciso. No entanto:
*   **Agência A (SP):** Tem dados de milhões de clientes, mas não pode enviá-los para a nuvem por sigilo bancário.
*   **Agência B (RJ):** Tem dados de fraudes recentes que a Agência A ainda não conhece.
*   **Agência C (DF):** Tem dados de comportamento de alto risco.
*   **Problema:** Como criar um modelo que aprenda com as 3 agências sem que nenhuma delas compartilhe seus dados brutos com as outras ou com um servidor central?
*   **Solução:** **Federated Learning**. O modelo vai até o dado, treina localmente e envia apenas os "ajustes" (pesos) de volta.

---

## 2. Setup do Ambiente
Vamos utilizar o **PyTorch** para manipular os pesos dos modelos manualmente.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

# Configurando o dispositivo (GPU se disponível)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" usando dispositivo: {device}")
```

---

## 3. Passo 1: Criando o Modelo Global e os Dados das Agências
Vamos simular um problema de classificação simples (Fraud vs. Legal). Criaremos 3 datasets diferentes, representando os dados privados de cada agência.

```python
# Definindo um modelo simples de rede neural (2 camadas)
class FraudModel(nn.Module):
    def __init__(self):
        super(FraudModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    def forward(self, x):
        return self.fc(x)

# Criando o Modelo Global (O mestre que coordena tudo)
global_model = FraudModel().to(device)

# Simulando Dados Privados das 3 Agências (10 features cada)
# Agência A: 1000 amostras | Agência B: 800 amostras | Agência C: 1200 amostras
data_a = torch.randn(1000, 10).to(device)
labels_a = torch.randint(0, 2, (1000,)).to(device)

data_b = torch.randn(800, 10).to(device)
labels_b = torch.randint(0, 2, (800,)).to(device)

data_c = torch.randn(1200, 10).to(device)
labels_c = torch.randint(0, 2, (1200,)).to(device)

agencies_data = [
    DataLoader(TensorDataset(data_a, labels_a), batch_size=32),
    DataLoader(TensorDataset(data_b, labels_b), batch_size=32),
    DataLoader(TensorDataset(data_c, labels_c), batch_size=32)
]
```

---

## 4. Passo 2: Treinamento Local (Cada Agência na sua Casa)
Nesta fase, enviamos uma cópia do modelo global para cada agência. Cada uma treina o modelo com seus próprios dados.

```python
def train_local(model, dataloader, epochs=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model.state_dict() # Retorna apenas os pesos (O Segredo Ajustado)

# Cada agência recebe o modelo global e treina localmente
local_weights = []
for i, loader in enumerate(agencies_data):
    print(f"📡 Enviando modelo global para Agência {i+1}...")
    model_copy = copy.deepcopy(global_model)
    weights = train_local(model_copy, loader)
    local_weights.append(weights)
    print(f"✅ Agência {i+1} concluiu o treino e devolveu os pesos ajustados.")
```

---

## 5. Passo 3: Agregação Global (Federated Averaging - FedAvg)
O servidor central recebe os pesos de todas as agências e calcula a **média aritmética** deles. O dado original nunca saiu da agência, apenas o conhecimento estatístico (pesos).

```python
# Pegando as chaves do dicionário de pesos
model_keys = local_weights[0].keys()
global_weights = global_model.state_dict()

# Média Simples (FedAvg)
for key in model_keys:
    # Soma os pesos de todas as agências para essa camada
    layer_sum = sum([weights[key] for weights in local_weights])
    # Calcula a média
    global_weights[key] = layer_sum / len(local_weights)

# Atualizando o Modelo Global com o conhecimento coletivo
global_model.load_state_dict(global_weights)
print("🧠 Modelo Global atualizado com sucesso via Federated Averaging!")
```

---

## 6. Desafio Tier 2: O Peso da Influência
Na vida real, agências com mais dados devem ter mais "voz" na média final.
1. Altere a lógica da agregação no Passo 3 para uma **Média Ponderada**.
2. Use os pesos: Agência A (33%), Agência B (27%), Agência C (40%).
3. **Reflexão:** O que acontece se uma agência enviar dados propositalmente errados (Ataque de Envenenamento)? O modelo global morre?

---

## 7. 💡 Gabarito do Desafio (Média Ponderada)

Se você implementou a **Média Ponderada**, seu código de agregação deve ser parecido com este:

```python
# Proporções de dados das agências
num_samples = [1000, 800, 1200]
total_samples = sum(num_samples)
proportions = [s / total_samples for s in num_samples]

# Agregação Ponderada
for key in model_keys:
    weighted_sum = sum([local_weights[i][key] * proportions[i] for i in range(len(local_weights))])
    global_weights[key] = weighted_sum
```

**Resposta sobre Envenenamento:** Sim, o modelo global é vulnerável. Em Federated Learning avançado, usamos técnicas de **Robust Aggregation** (como o algoritmo *Krum* ou *Median*) para ignorar atualizações de pesos que sejam muito discrepantes da maioria.

---

## 8. Entrega (Relatório de Aprendizado)
Responda às perguntas abaixo:

1.  **Privacidade:** Em algum momento a Agência A viu os dados da Agência B?
2.  **Banda:** O que ocupa mais rede: Enviar 1 milhão de fotos (dados) ou 1 arquivo de pesos (modelo)?
3.  **Conclusão:** Por que o Federated Learning é o futuro da IA em bancos como a CAIXA?
