# Lab 01b: O Custo da Conversa (Overhead, Resiliência e Filas)

**Objetivo:** Implementar a lógica de coordenação de um cluster e medir como a latência de rede, as falhas de hardware e a **Teoria das Filas** impactam o SLA de sistemas bancários massivos.
**Ambiente:** Google Colab.
**Cenário:** Planejamento de infraestrutura para o sistema de processamento de transações PIX (CAIXA).

---

## 🛠️ Passo 1: O Simulador de Rede Distribuída

Em sistemas distribuídos, cada vez que enviamos uma tarefa para outro servidor, pagamos um preço de rede (Latência).

```python
import time
import random
import numpy as np
import matplotlib.pyplot as plt

def enviar_e_processar(batch_size, latencia_ms, taxa_falha=0.01):
    """
    Simula o envio de um lote de transações para um worker.
    latencia_ms: Custo fixo de ida e volta (Round Trip Time).
    taxa_falha: Probabilidade do servidor worker falhar (exige re-envio).
    """
    time.sleep(latencia_ms / 1000)
    if random.random() < taxa_falha:
        return None # Falhou!
    tempo_cpu = batch_size * 0.001
    time.sleep(tempo_cpu)
    return True # Sucesso!
```

---

## 🚀 Passo 2: O Desafio do Batch Size (Tamanho do Lote)

**Exercício:** Implemente um loop que teste os tamanhos de lote `[1, 10, 50, 100, 500, 1000]` para processar 1.000 transações com latência de 100ms.

```python
total_tarefas = 1000
latencia = 100 # ms
batch_sizes = [1, 10, 50, 100, 500, 1000]
tempos_execucao = []

for b_size in batch_sizes:
    start_t = time.time()
    processadas = 0
    while processadas < total_tarefas:
        if enviar_e_processar(b_size, latencia):
            processadas += b_size
    tempos_execucao.append(time.time() - start_t)

plt.bar([str(b) for b in batch_sizes], tempos_execucao, color='orange')
plt.title('Impacto do Batch Size (Latência 100ms)'); plt.show()
```

---

## ⚡ Passo 3: O Paradoxo do Crescimento (Saturação)

Simulação de um cluster onde o custo de coordenação cresce conforme adicionamos máquinas.

```python
def simular_cluster(n_nos, total_tarefas, latencia_ms):
    tempo_calculo = (total_tarefas * 0.001) / n_nos
    overhead_rede = (latencia_ms / 1000) * n_nos
    return tempo_calculo + overhead_rede

nos = np.arange(1, 101)
resultados = [simular_cluster(n, 100000, 20) for n in nos]
plt.plot(nos, resultados, lw=3)
plt.axvline(nos[np.argmin(resultados)], color='red', linestyle='--')
plt.title('Identificando o Ponto de Saturação'); plt.show()
```

---

## 🚦 Passo 4: Teoria das Filas (O Gargalo Invisível)

Na vida real, as transações chegam em um fluxo constante. Se a taxa de chegada ($\lambda$) for maior que a capacidade de processamento do cluster ($\mu$), a fila cresce infinitamente.

**Conceito:** O tempo médio que uma transação espera na fila é dado por: $W = \frac{1}{\mu - \lambda}$

```python
def tempo_espera_fila(chegada, capacidade):
    if chegada >= capacidade:
        return 500 # Simula colapso/timeout
    return 1 / (capacidade - chegada)

taxas_chegada = np.linspace(10, 99, 100) # De 10 a 99 PIX/segundo
capacidade_cluster = 100 # O cluster aguenta 100 PIX/segundo

esperas = [tempo_espera_fila(c, capacidade_cluster) for c in taxas_chegada]

plt.figure(figsize=(10, 5))
plt.plot(taxas_chegada, esperas, color='purple', lw=2)
plt.title('Teorema das Filas: O "Joelho" da Curva')
plt.xlabel('Carga do Sistema (Requisições/Seg)')
plt.ylabel('Tempo de Espera na Fila (Segundos)')
plt.grid(True); plt.show()
```

### 🧠 Provocação 03:
**Pergunta:** Observe o gráfico acima. Por que o tempo de espera explode violentamente quando passamos de 90% de ocupação do cluster? O que acontece com o PIX da CAIXA se o cluster estiver operando no limite?

---

## 🔥 Desafio Extra (Tier 2 - Nível MLOps)

### Resiliência e Exponential Backoff
Implemente uma lógica de **Retry** com tempo de espera que dobra a cada falha (Backoff) e veja como isso ajuda a "desafogar" a fila do Passo 4 em momentos de instabilidade.

---

## 🏁 Conclusão de Engenharia
1. **Batching:** Esconde a latência.
2. **Saturação:** Limita a escala horizontal.
3. **Teoria das Filas:** Prova que rodar um cluster a 100% de CPU é uma estratégia perigosa (o tempo de resposta tende ao infinito).
