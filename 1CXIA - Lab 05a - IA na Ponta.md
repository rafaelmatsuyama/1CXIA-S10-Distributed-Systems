# Lab 05a: IA na Ponta (Edge Computing)

**Disciplina:** Sistemas Distribuídos para IA (1CXIA - S10)
**Objetivo:** Aprender a converter e otimizar modelos de IA para dispositivos de borda (Celulares, IoT) usando os formatos ONNX e TensorFlow Lite.
**Ambiente:** Google Colab (CPU/GPU) ou Databricks.

---

## 🚀 Instruções de Acesso ao Google Colab

Este laboratório pode ser realizado apenas com CPU, mas o uso de GPU acelera a inferência do TensorFlow original.

1.  Acesse: [https://colab.research.google.com/](https://colab.research.google.com/) e crie um **Novo Notebook**.
2.  (Opcional) **ATIVAR GPU:** Vá em `Ambiente de Execução` -> `Alterar o tipo de ambiente de execução` -> Selecione **T4 GPU** e clique em **Salvar**.

---

## 1. O Problema: Modelos Pesados em Dispositivos Leves
Um modelo treinado em um cluster de GPUs (como o A100 da Azure) é muito pesado para rodar no smartphone de um cliente da CAIXA ou em um caixa eletrônico antigo.
*   **Problema 1:** O arquivo do modelo é grande demais para download via 4G/5G.
*   **Problema 2:** O hardware local não suporta bibliotecas pesadas de treinamento (como o PyTorch completo).
*   **Solução:** Converter o modelo para formatos de "Somente Inferência" (Inference-only) como **ONNX** e **TFLite**, que são leves e rápidos.

---

## 2. Setup do Ambiente
Execute a célula abaixo para instalar as ferramentas de conversão:

```python
!pip install -q onnx tf2onnx tensorflow onnxruntime
```

---

## 3. Passo 1: O Modelo de Visão (MobileNetV2)
Vamos carregar um modelo famoso de classificação de imagens (`MobileNetV2`), projetado originalmente para dispositivos móveis, mas ainda em seu formato "pesado" de treinamento (Keras/TensorFlow).

```python
import tensorflow as tf
import os
import time

# Carregando o modelo MobileNetV2 pré-treinado na ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# Exportando o modelo (SavedModel) para compatibilidade com TFLite/ONNX
model.export("mobilenet_v2_original")

# Medindo o tamanho do modelo exportado
size_bytes = sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk("mobilenet_v2_original") for f in filenames)
print(f"📦 Tamanho do Modelo Original: {size_bytes / 1024**2:.2f} MB")
```

---

## 4. Passo 2: Conversão para TensorFlow Lite (TFLite)
O TFLite é o padrão ouro para Android e iOS. Ele remove metadados de treinamento e otimiza as operações matemáticas.

```python
# Criando o conversor TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("mobilenet_v2_original")

# Realizando a conversão
tflite_model = converter.convert()

# Salvando o arquivo .tflite
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print(f"✅ Conversão TFLite concluída!")
print(f"📉 Novo Tamanho (TFLite): {os.path.getsize('model.tflite') / 1024**2:.2f} MB")
```

---

## 5. Passo 3: Conversão para ONNX (Open Neural Network Exchange)
O ONNX é um formato universal que permite rodar modelos de qualquer framework (PyTorch, TF) em qualquer hardware (Nvidia, Intel, Qualcomm).

```python
import tf2onnx
import onnx

# Definindo a assinatura do modelo (Input Shape)
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

# Convertendo para ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Salvando o arquivo .onnx
with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"✅ Conversão ONNX concluída!")
print(f"🌍 Novo Tamanho (ONNX): {os.path.getsize('model.onnx') / 1024**2:.2f} MB")
```

---

## 6. Passo 4: O Teste de Velocidade (Inferência na Borda)
Vamos simular uma requisição de classificação e comparar o tempo de resposta entre o modelo original e o otimizado (usando o ONNX Runtime).

```python
import numpy as np
import onnxruntime as ort

# Criando um dado de teste aleatório (Uma imagem fake)
dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)

# 1. Teste com o Modelo Original (TensorFlow)
start = time.time()
res_tf = model.predict(dummy_input, verbose=0)
time_tf = time.time() - start

# 2. Teste com o Modelo Otimizado (ONNX Runtime)
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
start = time.time()
res_onnx = session.run(None, {input_name: dummy_input})
time_onnx = time.time() - start

print(f"🚀 Tempo Original (TensorFlow): {time_tf*1000:.2f} ms")
print(f"🔥 Tempo Otimizado (ONNX): {time_onnx*1000:.2f} ms")
print(f"📈 Ganho de Performance: {time_tf / time_onnx:.2f}x")
```

---

## 7. Desafio Tier 2: Quantização para Edge (INT8)
O TFLite permite "achatar" os pesos de FP32 para INT8, reduzindo o tamanho em mais 4x.
1. Adicione a linha `converter.optimizations = [tf.lite.Optimize.DEFAULT]` no código do Passo 2.
2. Reconverta o modelo e salve como `model_quantized.tflite`.
3. Qual o tamanho final do arquivo? Ele conseguiria rodar em um relógio inteligente (Smartwatch)?

### 💡 Gabarito do Desafio (Código de Quantização)

```python
# Criando o conversor TFLite com Otimização
converter_q = tf.lite.TFLiteConverter.from_saved_model("mobilenet_v2_original")

# ATIVANDO A QUANTIZAÇÃO (A mágica acontece aqui)
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]

# Realizando a conversão quantizada
tflite_quant_model = converter_q.convert()

# Salvando
with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

print(f"📉 Tamanho Quantizado (INT8): {os.path.getsize('model_quantized.tflize') / 1024**2:.2f} MB")
# Esperado: ~4 MB (Redução de 75% em relação ao TFLite original de ~14 MB)
```

---

## 8. Entrega (Tabela de Resultados)
Preencha os valores observados no seu ambiente:

| Formato | Tamanho (MB) | Tempo de Inferência (ms) | Facilidade de Deploy |
| :--- | :--- | :--- | :--- |
| **TensorFlow (SavedModel)** | | | Baixa (Requer Python/TF) |
| **TFLite (.tflite)** | | | Alta (Mobile/Web) |
| **ONNX (.onnx)** | | | Altíssima (Qualquer HW) |
