# Módulo 8: Frameworks y Desarrollo de Aplicaciones

## Contenido

### 8.1 LangChain Avanzado

**LCEL (LangChain Expression Language)**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Chain with LCEL
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Execute
result = chain.invoke({"topic": "AI"})

# Streaming
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Batch
results = chain.batch([{"topic": "AI"}, {"topic": "ML"}])

# Async
await chain.ainvoke({"topic": "AI"})
```

**LangGraph - Stateful Workflows**
```python
from langgraph.graph import StateGraph, END

# Define state
class AgentState(TypedDict):
    messages: list
    next: str

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", take_action)

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

app = workflow.compile()

# Execute
result = app.invoke({"messages": [("user", "Hello")]})
```

**LangServe - Deployment**
```python
from langserve import add_routes
from fastapi import FastAPI

app = FastAPI()

# Add chain as endpoint
add_routes(
    app,
    chain,
    path="/my-chain"
)

# Now available at:
# POST /my-chain/invoke
# POST /my-chain/stream
# POST /my-chain/batch
```

### 8.2 LlamaIndex

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is AI?")

# Chat engine
chat_engine = index.as_chat_engine()
response = chat_engine.chat("Tell me about transformers")

# Streaming
streaming_response = chat_engine.stream_chat("Explain RAG")
for token in streaming_response.response_gen:
    print(token, end="")
```

### 8.3 Semantic Kernel (Microsoft)

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# Initialize
kernel = sk.Kernel()

# Add service
kernel.add_service(
    OpenAIChatCompletion(
        ai_model_id="gpt-4",
        api_key="your-key"
    )
)

# Create function
@kernel.function(
    name="summarize",
    description="Summarizes text"
)
def summarize(text: str) -> str:
    return f"Summary of: {text}"

# Execute
result = await kernel.invoke(summarize, text="Long text...")
```

### 8.4 Haystack

```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator

# Build pipeline
pipeline = Pipeline()

pipeline.add_component("retriever", InMemoryBM25Retriever(document_store))
pipeline.add_component("generator", OpenAIGenerator())

pipeline.connect("retriever.documents", "generator.documents")

# Run
result = pipeline.run({
    "retriever": {"query": "What is AI?"},
    "generator": {"query": "What is AI?"}
})
```

### 8.5 Arquitecturas de Producción

**Microservices Architecture**
```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   API       │────▶│   LLM       │────▶│   Vector     │
│   Gateway   │     │   Service   │     │   DB Service │
└─────────────┘     └─────────────┘     └──────────────┘
      │                    │                     │
      │                    ▼                     ▼
      │             ┌─────────────┐     ┌──────────────┐
      │             │  Embedding  │     │   Document   │
      │             │  Service    │     │   Store      │
      │             └─────────────┘     └──────────────┘
      ▼
┌─────────────┐
│  Monitoring │
│  & Logging  │
└─────────────┘
```

**Caching Layer**
```python
from functools import lru_cache
import hashlib
import redis

class LLMCache:
    def __init__(self):
        self.redis_client = redis.Redis()

    def get_or_generate(self, prompt: str, llm_func):
        # Create cache key
        key = hashlib.md5(prompt.encode()).hexdigest()

        # Check cache
        cached = self.redis_client.get(key)
        if cached:
            return cached.decode()

        # Generate and cache
        result = llm_func(prompt)
        self.redis_client.setex(key, 3600, result)  # 1 hour TTL

        return result
```

**Load Balancing**
```python
import random

class LLMLoadBalancer:
    def __init__(self, endpoints: list):
        self.endpoints = endpoints
        self.current = 0

    def round_robin(self):
        endpoint = self.endpoints[self.current]
        self.current = (self.current + 1) % len(self.endpoints)
        return endpoint

    def least_loaded(self):
        # Get endpoint with lowest load
        loads = [self.get_load(ep) for ep in self.endpoints]
        return self.endpoints[loads.index(min(loads))]
```

## Ejercicios

1. Crear pipeline RAG completo con LangChain
2. Implementar sistema multi-agente con LangGraph
3. Desplegar aplicación con LangServe
4. Diseñar arquitectura escalable para producción

[← Módulo Anterior](../modulo07_open_source_deployment/) | [Siguiente →](../modulo09_guardrails_seguridad/)
