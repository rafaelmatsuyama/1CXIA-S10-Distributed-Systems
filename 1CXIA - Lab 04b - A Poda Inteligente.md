# Lab 04b: A Poda Inteligente (Pruning & Batching)

**Disciplina:** Sistemas Distribuídos para IA (1CXIA - S10)
**Objetivo:** Aplicar técnicas de Poda (Pruning) para reduzir o tamanho de modelos e Loteamento (Batching) para maximizar o Throughput.
**Ambiente:** Google Colab (Exclusivo).

---

## 🚀 Instruções de Acesso ao Google Colab

Este laboratório utiliza GPU para aceleração de inferência. Siga os passos:

1.  Acesse: [https://colab.research.google.com/](https://colab.research.google.com/) e crie um **Novo Notebook**.
2.  **ATIVAR GPU:** Vá em `Ambiente de Execução` -> `Alterar o tipo de ambiente de execução`.
3.  Selecione **T4 GPU** e clique em **Salvar**.

---

## 1. O Conceito: Menos é Mais (Pruning) e Lotes (Batching)
Em Sistemas Distribuídos de IA, temos dois grandes desafios:
*   **Armazenamento/Memória:** O modelo precisa caber em dispositivos com recursos limitados (Edge/Agências).
*   **Vazão (Throughput):** O servidor precisa processar o máximo de requisições simultâneas possível.

Neste laboratório, vamos "podar" uma rede de visão computacional (**ResNet-18**) e comparar a velocidade de processamento individual vs. coletivo.

---

## 2. Setup e Baseline do Modelo
Vamos carregar uma ResNet-18 pré-treinada e verificar seu tamanho original.

```python
import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import time
import os

# Carregando ResNet-18 original
model = models.resnet18(weights="IMAGENET1K_V1").to("cuda")

# Verificando tamanho original em disco (Simulado via salvamento)
torch.save(model.state_dict(), "resnet18_original.pth")
size_orig = os.path.getsize("resnet18_original.pth") / (1024**2)
print(f"💾 Tamanho Original: {size_orig:.2f} MB")
```

---

## 3. Passo 1: A Poda (Unstructured Pruning)
Vamos remover **50% dos pesos** das camadas convolucionais que possuem os menores valores absolutos (L1 Unstructured).

```python
# Aplicando poda de 50% em todas as camadas convolucionais
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.5)
        prune.remove(module, 'weight') # Torna a poda permanente

# Verificando a esparsidade (Quantos pesos agora são ZERO)
total_zeros = 0
total_params = 0
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        total_zeros += torch.sum(module.weight == 0)
        total_params += module.weight.nelement()

print(f"✂️ Esparsidade: {100. * float(total_zeros) / total_params:.2f}% dos neurônios removidos.")
```

---

## 4. Passo 2: Batching (Processamento Individual vs. Lote)
Aqui está o segredo do Throughput em sistemas distribuídos. Vamos processar 64 imagens de duas formas.

```python
# Simulando 64 imagens (3 canais, 224x224 pixels)
data = torch.randn(64, 3, 224, 224).to("cuda")

# TESTE A: Processamento Individual (Batch Size = 1)
start_a = time.time()
for i in range(64):
    _ = model(data[i].unsqueeze(0))
end_a = time.time()
latencia_individual = (end_a - start_a) / 64

# TESTE B: Processamento em Lote (Batch Size = 64)
start_b = time.time()
_ = model(data)
end_b = time.time()
latencia_lote = (end_b - start_b) / 64

print(f"⏱️ Tempo Médio (Individual): {latencia_individual:.4f}s")
print(f"⏱️ Tempo Médio (Lote): {latencia_lote:.4f}s")
print(f"🚀 Ganho de Throughput (Vazão): {latencia_individual / latencia_lote:.2f}x")
```

---

## 5. Desafio Tier 2: A Poda Radical
1.  Aumente a poda para **90%** (`amount=0.9`).
2.  Tente salvar o modelo com compressão (usando `zip` ou salvando apenas pesos não nulos) e veja se o tamanho do arquivo realmente diminui na prática (o PyTorch salva tensores esparsos de forma densa por padrão).
3.  **Reflexão:** Por que o Batching é mais rápido se o modelo é o mesmo? (Dica: Paralelismo na GPU).

---

## 6. 🔑 Gabarito do Desafio (Resultado Esperado)

Se você realizou a **Poda Radical (90%)**, deve ter observado os seguintes fenômenos:

1.  **Acurácia:** O modelo ainda gera saídas válidas (não trava), mas o reconhecimento de imagem falha drasticamente. Com 90% dos pesos zerados, o modelo perde as "características finas" das imagens.
2.  **Velocidade (O Pulo do Gato):** No PyTorch padrão, **a velocidade NÃO muda com a poda**. Por quê? Porque o PyTorch continua multiplicando matrizes densas (cheias de zeros). Para ganhar velocidade real, seriam necessárias bibliotecas de **Cálculo Esparso (Sparse Kernels)** que ignoram os zeros.
3.  **Batching:** A vazão aumenta no lote porque a GPU é um **processador massivamente paralelo**. Processar uma imagem por vez subutiliza os milhares de núcleos da GPU. No lote, a GPU preenche todos os seus núcleos e processa tudo de uma vez.

---

## 7. Entrega (Tabela de Resultados)
Preencha os valores observados no seu ambiente:

| Técnica | Resultado Observado | Impacto Esperado |
| :--- | :--- | :--- |
| **Esparsidade (%)** | | Redução de complexidade |
| **Ganho de Vazão (x)** | | Eficiência Distribuída |
| **Tamanho Original** | | 45 MB |
