# Lab 01a: A Ilusão da Velocidade (Speedup)

**Objetivo:** Medir o ganho real de performance ao paralelizar uma tarefa de processamento intensivo de CPU/GPU e entender a necessidade vital de clusters para cálculos de risco bancário.
**Ambiente:** Google Colab.
**Cenário:** Simulação de Monte Carlo para Cálculo de Risco em 500 Propostas de Crédito de Alto Valor (CAIXA).

---

## 🚀 Passo 0: Preparação do Ambiente (Google Colab)

1.  Acesse o link: [https://colab.research.google.com/](https://colab.research.google.com/)
2.  Clique em **"Novo Notebook"**.
3.  **Configuração da GPU:**
    *   No menu superior, vá em **Ambiente de execução** > **Alterar tipo de ambiente de execução**.
    *   Em "Acelerador de hardware", selecione **T4 GPU**.
    *   Clique em **Salvar**.
4.  Verifique se o ambiente está pronto rodando a célula: `!nvidia-smi` (Deve listar uma GPU Tesla T4).

---

## 🛠️ Passo 1: O Cenário (Execução Serial - CPU)

Vamos simular o **Risco de Crédito**. Para cada proposta de empréstimo, realizaremos **20.000.000 (20 milhões)** de simulações de cenários econômicos. Esta é uma tarefa puramente **CPU Bound**.

```python
import time
import numpy as np
import torch
import os

# Simulação de 500 propostas de alto valor
propostas = [{"id": i, "valor": np.random.randint(100000, 1000000)} for i in range(500)]

def simular_risco_monte_carlo(proposta):
    # 20.000.000 de iterações garante ~6 a 8 minutos de execução serial no Colab
    n_cenarios = 20000000 
    
    # Geração de cenários de inadimplência
    cenarios = np.random.normal(0.05, 0.15, n_cenarios)
    
    # Cálculo de Perda com Desconto de Valor Presente (NPV)
    perdas_brutas = proposta['valor'] * (cenarios[cenarios > 0.20])
    taxas_desconto = np.random.uniform(0.08, 0.12, len(perdas_brutas))
    perdas_descontadas = perdas_brutas / (1 + taxas_desconto)
    
    return np.mean(perdas_descontadas) if len(perdas_descontadas) > 0 else 0

# Execução Serial (Baseline)
print(f"Iniciando simulação serial para {len(propostas)} propostas... Isso levará aprox. 6-7 minutos.")
start_time = time.time()
resultados_seriais = [simular_risco_monte_carlo(p) for p in propostas]
tempo_serial = time.time() - start_time

print(f"--- Execução Serial Finalizada ---")
print(f"Tempo Total: {tempo_serial:.2f} segundos (~{tempo_serial/60:.1f} min)")
```

### 🧠 Provocação 01 (Cenário Serial):
**Pergunta:** Se a CAIXA recebe **500.000 contratos** para análise em um único dia, quanto tempo levaria para processar tudo utilizando apenas 1 Core de CPU com este script? Faça o cálculo matemático baseado no seu resultado.

---

## 🚀 Passo 2: Aceleração Paralela (Multi-CPU)

Utilizaremos o `ProcessPoolExecutor` para distribuir as propostas entre os núcleos da CPU do Google Colab.

```python
from concurrent.futures import ProcessPoolExecutor

n_cores = os.cpu_count()
print(f"Núcleos disponíveis: {n_cores}")

print(f"Iniciando simulação paralela em {n_cores} núcleos...")
start_time = time.time()

with ProcessPoolExecutor(max_workers=n_cores) as executor:
    resultados_paralelos = list(executor.map(simular_risco_monte_carlo, propostas))

tempo_paralelo = time.time() - start_time
print(f"--- Execução Paralela (CPU) Finalizada ---")
print(f"Tempo Total: {tempo_paralelo:.2f} segundos")
```

### 🧠 Provocação 02 (Cenário Paralelo):
**Pergunta:** O seu ganho foi de exatamente 2x (se o Colab te deu 2 cores)? Se não, onde o tempo foi "perdido"? Pense no custo de comunicação entre processos (**IPC**).

---

## ⚡ Passo 3: O Salto Tecnológico (Aceleração por GPU)

Vamos usar o **PyTorch** para rodar a simulação de todos os contratos simultaneamente nos **2.560 núcleos CUDA** da GPU T4.

```python
def simular_gpu(propostas_lista):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando simulação em GPU ({device}) via PyTorch...")
    
    valores = torch.tensor([p['valor'] for p in propostas_lista], device=device).view(-1, 1)
    n_cenarios_gpu = 1000000 # 1 Milhão por proposta para evitar estouro de VRAM
    n_propostas = len(propostas_lista)
    
    start_time_gpu = time.time()
    
    # Processamento Vetorial Massivo
    cenarios = torch.randn((n_propostas, n_cenarios_gpu), device=device) * 0.15 + 0.05
    mask = cenarios > 0.20
    perdas_brutas = valores * mask.float()
    resultados_gpu = perdas_brutas.mean(dim=1)
    
    torch.cuda.synchronize() # Sincroniza para medição real de tempo
    tempo_final_gpu = time.time() - start_time_gpu
    return resultados_gpu, tempo_final_gpu

res_gpu, tempo_gpu = simular_gpu(propostas)
print(f"--- Execução GPU Finalizada ---")
print(f"Tempo Total: {tempo_gpu:.2f} segundos")
```

### 🧠 Provocação 03 (Cenário GPU):
**Pergunta:** A GPU foi centenas de vezes mais rápida. Por que ainda usamos CPUs para processamento distribuído (como no Spark)? O que acontece se o dado for maior que a memória da GPU (**16GB**)?

---

## 📊 Passo 4: Comparativo Final

```python
print(f"--- QUADRO COMPARATIVO ---")
print(f"1. Serial (CPU):   {tempo_serial:.2f}s")
print(f"2. Paralelo (CPU): {tempo_paralelo:.2f}s (Speedup: {tempo_serial/tempo_paralelo:.2f}x)")
print(f"3. GPU (PyTorch):  {tempo_gpu:.2f}s (Speedup: {tempo_serial/tempo_gpu:.2f}x)")
```

---

## 💡 Guia de Troubleshooting (Engenheiro de MLOps)
*   **Vetorização:** A GPU processa uma **Matriz** inteira (SIMD).
*   **Amdahl:** O limite de velocidade é ditado pela parte do código que não pode ser paralelizada.
*   **VRAM:** O limite físico da placa de vídeo exige estratégias de **Sharding** (próximas aulas).
