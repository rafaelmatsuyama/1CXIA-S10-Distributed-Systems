# Lab 04a: O Peso da Precisão (Quantização)

**Disciplina:** Sistemas Distribuídos para IA (1CXIA - S10)
**Objetivo:** Medir o impacto da precisão numérica (FP32 vs. INT4) na performance de modelos de linguagem (LLMs).
**Ambiente:** Google Colab (Exclusivo).

---

## 🚀 Instruções de Acesso ao Google Colab

Este laboratório requer uma GPU para funcionar, devido à biblioteca `bitsandbytes`. Siga os passos abaixo:

1.  Acesse: [https://colab.research.google.com/](https://colab.research.google.com/) e crie um **Novo Notebook**.
2.  **ATIVAR GPU:** Vá em `Ambiente de Execução` -> `Alterar o tipo de ambiente de execução`.
3.  Selecione **T4 GPU** e clique em **Salvar**.
4.  Confirme se o ícone verde no canto superior direito mostra **"T4"**.

---

## 1. O Problema: O Custo da Precisão Total
Modelos de IA modernos, como o Llama ou BERT, são treinados em **FP32** (32-bit Floating Point). Cada parâmetro do modelo ocupa 4 bytes de memória. 
*   Um modelo de **7 Bilhões de parâmetros** em FP32 ocuparia **~28 GB** de VRAM. 
*   Isso inviabiliza o uso em hardware comum ou aumenta drasticamente o custo na nuvem.

Neste laboratório, vamos aplicar a **Quantização de 4-bits** (NF4) para reduzir esse custo em **8x** e aumentar a velocidade de resposta.

---

## 2. Setup do Ambiente
Execute a célula abaixo para instalar as bibliotecas de otimização:

```python
!pip install -q -U bitsandbytes transformers accelerate
```

---

## 3. Passo 1: O Modelo em Precisão Total (Baseline)
Vamos carregar o modelo `TinyLlama-1.1B` em sua forma original. Mesmo sendo um modelo pequeno, veremos o consumo de memória.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Carregando Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Carregando Modelo em FP32 (Original)
start_time = time.time()
model_fp32 = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    device_map="auto"
)
load_time_fp32 = time.time() - start_time

print(f"✅ Modelo FP32 carregado em {load_time_fp32:.2f} segundos.")
print(f"💾 Memória Ocupada (VRAM): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

---

## 4. Passo 2: O Benchmark de Velocidade
Vamos gerar uma resposta e medir quantos **tokens por segundo** o sistema consegue entregar.

```python
messages = [{"role": "user", "content": "Explique o que é um sistema distribuído em 3 frases."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

# Benchmark FP32
start_gen = time.time()
output = model_fp32.generate(
    **input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)
end_gen = time.time()

# Calculando apenas os novos tokens gerados
new_tokens = output[0][input_ids.input_ids.shape[-1]:]
tps_fp32 = len(new_tokens) / (end_gen - start_gen)

print(f"🚀 Performance FP32: {tps_fp32:.2f} tokens/seg (Geração)")
print(f"📝 Resposta: {tokenizer.decode(new_tokens, skip_special_tokens=True)}")

# Limpando memória para o próximo passo
del model_fp32
torch.cuda.empty_cache()
```

---

## 5. Passo 3: A Dieta (Quantização 4-bit)
Agora, vamos carregar o **mesmo modelo**, mas forçando cada parâmetro a ocupar apenas **4 bits** usando a tecnologia `bitsandbytes`.

```python
from transformers import BitsAndBytesConfig

# Configuração de Quantização 4-bit (NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Carregando Modelo Quantizado
start_time = time.time()
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    device_map="auto"
)
load_time_4bit = time.time() - start_time

print(f"✅ Modelo 4-bit carregado em {load_time_4bit:.2f} segundos.")
print(f"💾 Memória Ocupada (VRAM): {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

---

## 6. Passo 4: O Benchmark Final (Comparação)
Repita o teste de geração e observe o ganho de Throughput.

```python
# Benchmark 4-bit
start_gen = time.time()
output = model_4bit.generate(
    **input_ids, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.7
)
end_gen = time.time()

# Calculando apenas os novos tokens gerados
new_tokens_4b = output[0][input_ids.shape[-1]:]
tps_4bit = len(new_tokens_4b) / (end_gen - start_gen)

print(f"🔥 Performance 4-bit: {tps_4bit:.2f} tokens/seg (Geração)")
print(f"📈 Ganho de Velocidade: {tps_4bit / tps_fp32:.2f}x")
print(f"📝 Resposta: {tokenizer.decode(new_tokens_4b, skip_special_tokens=True)}")
```

---

## 7. Desafio Tier 2: A Fronteira da Alucinação
A quantização remove detalhes matemáticos do modelo. 
1.  Faça uma pergunta lógica complexa (ex: "Se João tem 3 maçãs e perdeu 2, mas ganhou o dobro do que tinha no início, quantas ele tem agora?") para o modelo quantizado.
2.  Verifique se ele ainda consegue raciocinar corretamente ou se a "dieta" afetou a inteligência do sistema.

---

## 8. Entrega (Tabela de Resultados)
Preencha os valores observados no seu ambiente:

| Métrica | FP32 (Original) | INT4 (Quantizado) | Impacto |
| :--- | :--- | :--- | :--- |
| **Memória (VRAM)** | | | |
| **Tokens / Seg** | | | |
| **Qualidade da Resposta** | | | |
