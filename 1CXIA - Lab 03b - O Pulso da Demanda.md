# Lab 03b: O Pulso da Demanda (Auto-scaling e Deploy)

**Disciplina:** Sistemas Distribuídos para IA (1CXIA - S10)
**Contexto:** Orquestração de Containers (AKS) e Resiliência
**Duração Estimada:** 75 minutos

---

## 🎯 Objetivo do Lab
Provar que uma arquitetura baseada em microserviços (Kubernetes) é capaz de sobreviver a picos massivos de demanda e se auto-recuperar de falhas críticas ("Self-healing").

## 🏦 Cenário CAIXA
É dia de pagamento do **Bolsa Família**. O serviço de IA responsável pela validação biométrica no app está recebendo 10x mais requisições que o normal. Se o sistema cair, as agências ficarão lotadas. Precisamos garantir que o **AKS** crie novas réplicas do modelo automaticamente conforme a carga aumenta.

---

## 🛠️ Passo a Passo (Caminho Fast-Track)

### 🚀 1. Acesso ao Ambiente Kubernetes (Killercoda)
Para focar na orquestração e no comportamento da IA, utilizaremos um ambiente Kubernetes pronto que ignora restrições de cota da Azure.

1.  Acesse: [Killercoda Kubernetes Playground](https://killercoda.com/playgrounds/scenario/kubernetes)
2.  Aguarde o terminal carregar (levará ~5 segundos).
3. Tudo pronto! Você já está no controle de um cluster Kubernetes real.

### 📡 1.5 Ativando os Sensores (Metrics Server)
O Kubernetes precisa de "sensores" para ler o uso de CPU. Se este passo for ignorado, o Auto-scaling não funcionará.

1.  **Instale o servidor de métricas:**
    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    ```
2.  **Aplique o Patch de Segurança (Importante):**
    ```bash
    kubectl patch deployment metrics-server -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'
    ```
3.  **Valide:** Aguarde 30 segundos e rode `kubectl top nodes`. Se aparecer o uso de CPU/Memória, os sensores estão online!

---

### 📦 2. O Deploy do Modelo (A API de IA)

No terminal (seja no Killercoda ou no seu terminal configurado), vamos subir um container que simula uma API de IA. Criaremos um manifesto `deploy.yaml`:

```bash
cat <<EOF > deploy.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ia-biometria
spec:
  replicas: 1
  selector:
    matchLabels:
      app: biometria
  template:
    metadata:
      labels:
        app: biometria
    spec:
      containers:
      - name: api-ia
        image: registry.k8s.io/hpa-example
        resources:
          requests:
            cpu: 200m
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: biometria-service
spec:
  type: NodePort
  ports:
  - port: 80
    nodePort: 30001
  selector:
    app: biometria
EOF

kubectl apply -f deploy.yaml
```

### 📈 3. Configurando o HPA (O "Pulso")
Agora, definimos a regra de escalabilidade: se o uso de CPU passar de 50%, o Kubernetes deve criar novas cópias (Pods) da nossa IA.

```bash
kubectl autoscale deployment ia-biometria --cpu=50 --min=1 --max=10
```

---

## 🧪 Cenário de Caos: A Guerra de Acessos
Vamos simular um ataque de requisições para ver o sistema "pulsar".

1. Em um terminal separado (no Killercoda, clique no ícone `+` e `New Terminal`), monitore os Pods:
   `kubectl get pods -w`
2. No primeiro terminal, dispare a carga (Simulação massiva):
   `kubectl run -i --tty --rm load-generator --image=busybox:1.28 --restart=Never -- /bin/sh -c "while true; do wget -q -O- http://biometria-service; done"`
3. **Observação:** Em poucos segundos, você verá novos Pods surgindo (`ContainerCreating`) para dividir a carga. O sistema está "respirando"!

---

## 🏆 Desafio Tier 2 (O Golpe de Misericórdia)
O que acontece se um servidor inteiro "quebrar"?
*   **Tarefa:** Force a exclusão do Pod principal enquanto a carga está alta: `kubectl delete pod [nome-do-pod]`.
*   **Discussão:** O serviço parou? Quanto tempo o Kubernetes levou para criar um substituto?

---

## 🧠 Gabarito e Discussão
*   **Horizontal Pod Autoscaler (HPA):** Escalabilidade a nível de software (Pods). É o que salva a biometria da CAIXA em picos de acesso.
*   **Self-healing:** O Kubernetes não "conserta" um container quebrado; ele o descarta e cria um novo em milissegundos.

---

## 🚩 Opção B (Avançado): Provisionamento no Azure AKS
Use este guia apenas se sua assinatura Azure tiver cotas liberadas. 

```bash
# Comandos para criar o terreno na Azure (East US)
ID=7904; RG="rg-1cxia-caixa-$ID"; LOC="eastus"; AKS="aks-caixa-prod"

# Garantir o RG
az group create --name $RG --location $LOC

# Criar o AKS usando Standard_DS2_v2 (Standard Lab VM)
# Para contas pagas, o Cluster Autoscaler permite ver o hardware escalando.
az aks create \
    --resource-group $RG \
    --name $AKS \
    --node-count 1 \
    --node-vm-size Standard_DS2_v2 \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 3 \
    --generate-ssh-keys \
    --no-wait
```

> [!TIP]
> **DICA PARA O INSTRUTOR:** Se quiser descobrir quais máquinas você PODE criar na Azure, rode:
> `az vm list-usage --location eastus --query "[?limit>'0'].{Name:localName, Capacity:limit, Usage:currentValue}" --output table`
