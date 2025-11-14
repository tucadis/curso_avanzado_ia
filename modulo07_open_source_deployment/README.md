# Módulo 7: Modelos Open Source y Deployment

## Contenido

### 7.1 Ecosistema Open Source 2025

**Llama 4 (Meta)**
- 8B, 70B, 405B parámetros
- Multilingual (100+ idiomas)
- Licencia permisiva
- State-of-the-art performance

**Mistral Large 3**
- 123B parámetros
- Mejor que GPT-4 en coding
- Apache 2.0 license
- MoE architecture

**Qwen 3**
- 7B, 14B, 72B
- Excelente en chino e inglés
- Strong math/coding
- Free for commercial use

### 7.2 Hugging Face Ecosystem

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load any model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# Or use pipeline
pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.3",
    device_map="auto"
)

output = pipe("Explain quantum computing", max_new_tokens=200)
```

**Inference Endpoints**
```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="your_token")

# Deploy model
endpoint = client.create_inference_endpoint(
    name="my-llama-endpoint",
    repository="meta-llama/Llama-3-8B",
    framework="pytorch",
    task="text-generation",
    accelerator="gpu",
    instance_size="medium"
)

# Use endpoint
response = client.post(
    json={"inputs": "Hello, how are you?"},
    endpoint=endpoint.url
)
```

### 7.3 Optimización de Inferencia

**vLLM - Ultra Fast Inference**
```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=2,  # Multi-GPU
    gpu_memory_utilization=0.9
)

# Sampling params
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200
)

# Generate (batched for speed!)
prompts = ["Hello", "Explain AI", "Write code"]
outputs = llm.generate(prompts, sampling_params)

# 10-30x faster than HF!
```

**TensorRT-LLM**
```python
# Build optimized engine
trtllm-build \
    --checkpoint_dir ./llama-8b \
    --output_dir ./llama-8b-engine \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --use_fused_mlp \
    --use_gpt_attention_plugin

# Serve with Triton
docker run --gpus all \
    -v ./llama-8b-engine:/models \
    nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3
```

### 7.4 Ollama - Local Models

```bash
# Install
curl -fsSL https://ollama.ai/install.sh | sh

# Run models
ollama run llama3:8b
ollama run codellama:34b
ollama run mistral:7b

# API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Explain AI"
}'
```

```python
# Python client
import ollama

response = ollama.generate(
    model='llama3:8b',
    prompt='What is machine learning?'
)

print(response['response'])
```

### 7.5 Deployment en Producción

**Docker Deployment**
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install dependencies
RUN pip install vllm transformers

# Copy model
COPY ./model /app/model

# Serve
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model", \
     "--port", "8000"]
```

**Kubernetes**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-3-8B"
```

**FastAPI Wrapper**
```python
from fastapi import FastAPI
from vllm import LLM

app = FastAPI()
llm = LLM("meta-llama/Llama-3-8B")

@app.post("/generate")
async def generate(prompt: str):
    output = llm.generate(prompt)
    return {"response": output[0].outputs[0].text}
```

## Ejercicios

1. Desplegar Llama 3 con vLLM y benchmark vs HF
2. Crear API REST para modelo con FastAPI
3. Optimizar modelo con TensorRT
4. Configurar Ollama para uso local

[← Módulo Anterior](../modulo06_fine_tuning/) | [Siguiente →](../modulo08_frameworks_desarrollo/)
