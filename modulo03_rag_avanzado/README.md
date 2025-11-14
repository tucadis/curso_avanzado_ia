# Módulo 3: RAG Avanzado y Agentic RAG

## Descripción

Retrieval-Augmented Generation (RAG) ha evolucionado desde simples búsquedas vectoriales hasta sistemas sofisticados con razonamiento agéntico. Este módulo cubre las técnicas más avanzadas de RAG en 2025.

## Objetivos

- Implementar sistemas RAG avanzados con múltiples estrategias de recuperación
- Dominar GraphRAG y retrieval basado en grafos de conocimiento
- Construir sistemas Agentic RAG con toma de decisiones dinámica
- Optimizar embeddings y chunking strategies
- Evaluar sistemas RAG con métricas especializadas

## Contenido

### 3.1 RAG Fundamentals Evolution

**Pipeline RAG Básico**
```python
# Pipeline tradicional
1. Chunking: Dividir documentos
2. Embedding: Convertir a vectores
3. Indexing: Almacenar en vector DB
4. Query: Usuario hace pregunta
5. Retrieval: Buscar chunks relevantes
6. Augmentation: Añadir contexto al prompt
7. Generation: LLM genera respuesta
```

**Limitaciones del RAG Básico**
- Retrieval naive (solo similaridad)
- Sin re-ranking
- Chunks de tamaño fijo
- Sin validación de relevancia
- No maneja queries complejas

### 3.2 Técnicas Avanzadas de RAG

#### Hybrid Search

Combina búsqueda vectorial (semántica) con keyword search (BM25):

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

# Vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# BM25 keyword retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Ensemble (weighted combination)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # 40% keyword, 60% semantic
)

results = ensemble_retriever.get_relevant_documents(query)
```

#### Re-ranking

Mejora resultados usando un modelo de re-ranking:

```python
from sentence_transformers import CrossEncoder

# Re-ranker model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_documents(query: str, documents: List[str], top_k: int = 5):
    # Score each doc
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)

    # Sort by score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

#### Query Expansion

Expandir query para mejor recuperación:

```python
def expand_query(original_query: str, llm) -> List[str]:
    prompt = f"""
    Genera 3 variaciones de esta pregunta que puedan ayudar a encontrar información relevante:
    Pregunta original: {original_query}

    Variaciones:
    1.
    2.
    3.
    """

    response = llm(prompt)
    variations = parse_variations(response)

    return [original_query] + variations

# Buscar con todas las variaciones
all_queries = expand_query(user_query, llm)
all_results = []
for query in all_queries:
    results = retriever.get_relevant_documents(query)
    all_results.extend(results)

# Deduplicate and rerank
unique_results = deduplicate(all_results)
final_results = rerank_documents(user_query, unique_results)
```

#### Self-RAG

El modelo decide cuándo recuperar información:

```python
class SelfRAG:
    def generate(self, query: str):
        # 1. Decide if retrieval is needed
        needs_retrieval = self.check_if_retrieval_needed(query)

        if needs_retrieval:
            # 2. Retrieve documents
            docs = self.retriever.get_relevant_documents(query)

            # 3. Check relevance
            relevant_docs = self.filter_relevant(query, docs)

            # 4. Generate with context
            response = self.generate_with_context(query, relevant_docs)

            # 5. Verify response is supported by docs
            if self.is_supported(response, relevant_docs):
                return response
            else:
                # Retrieve more or regenerate
                return self.retry_generation(query)
        else:
            # Generate without retrieval
            return self.llm.generate(query)
```

#### Corrective RAG (CRAG)

Corrige retrievals de baja calidad:

```python
def corrective_rag(query: str):
    # 1. Initial retrieval
    docs = retriever.get_relevant_documents(query)

    # 2. Evaluate quality
    relevance_scores = evaluate_relevance(query, docs)

    # 3. Decide action
    if all(score > 0.8 for score in relevance_scores):
        # High quality - use directly
        return generate_answer(query, docs)

    elif any(score > 0.5 for score in relevance_scores):
        # Mixed quality - filter and augment
        good_docs = [d for d, s in zip(docs, relevance_scores) if s > 0.5]
        web_results = web_search(query)  # Augment with web
        return generate_answer(query, good_docs + web_results)

    else:
        # Poor quality - use alternative source
        return web_search_answer(query)
```

### 3.3 GraphRAG

Usa grafos de conocimiento para mejor contexto:

```python
from neo4j import GraphDatabase
from langchain.graphs import Neo4jGraph

class GraphRAG:
    def __init__(self, uri, user, password):
        self.graph = Neo4jGraph(url=uri, username=user, password=password)

    def create_knowledge_graph(self, documents):
        for doc in documents:
            # Extract entities and relationships
            entities = extract_entities(doc)
            relationships = extract_relationships(doc)

            # Add to graph
            for entity in entities:
                self.graph.query(
                    "MERGE (e:Entity {name: $name, type: $type})",
                    {"name": entity.name, "type": entity.type}
                )

            for rel in relationships:
                self.graph.query(
                    """
                    MATCH (a:Entity {name: $source})
                    MATCH (b:Entity {name: $target})
                    MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
                    """,
                    {"source": rel.source, "target": rel.target, "rel_type": rel.type}
                )

    def retrieve_with_graph(self, query: str):
        # 1. Extract entities from query
        query_entities = extract_entities(query)

        # 2. Find related entities in graph
        cypher_query = """
        MATCH (e:Entity {name: $entity})-[r*1..2]-(related)
        RETURN e, r, related
        LIMIT 20
        """

        graph_context = []
        for entity in query_entities:
            results = self.graph.query(cypher_query, {"entity": entity.name})
            graph_context.extend(results)

        # 3. Also do vector search
        vector_results = self.vector_retriever.get_relevant_documents(query)

        # 4. Combine context
        full_context = self.combine_context(graph_context, vector_results)

        return full_context
```

### 3.4 Agentic RAG

Agentes que deciden dinámicamente cómo recuperar:

```python
from langgraph.graph import StateGraph, END

class AgenticRAG:
    def __init__(self):
        self.workflow = self.build_workflow()

    def build_workflow(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("query_analyzer", self.analyze_query)
        workflow.add_node("router", self.route_query)
        workflow.add_node("vector_search", self.vector_search)
        workflow.add_node("graph_search", self.graph_search)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("synthesizer", self.synthesize_answer)
        workflow.add_node("validator", self.validate_answer)

        # Add edges
        workflow.set_entry_point("query_analyzer")

        workflow.add_conditional_edges(
            "router",
            self.decide_search_method,
            {
                "vector": "vector_search",
                "graph": "graph_search",
                "web": "web_search",
                "multi": ["vector_search", "graph_search"]
            }
        )

        workflow.add_edge("vector_search", "synthesizer")
        workflow.add_edge("graph_search", "synthesizer")
        workflow.add_edge("web_search", "synthesizer")

        workflow.add_conditional_edges(
            "validator",
            self.check_quality,
            {
                "good": END,
                "retry": "router",
                "insufficient": "web_search"
            }
        )

        return workflow.compile()

    def analyze_query(self, state):
        """Analiza la complejidad y tipo de query"""
        query = state["query"]

        analysis = {
            "complexity": self.assess_complexity(query),
            "type": self.classify_query_type(query),
            "entities": self.extract_entities(query),
            "requires_current_info": self.needs_current_data(query)
        }

        state["analysis"] = analysis
        return state

    def route_query(self, state):
        """Decide qué método de búsqueda usar"""
        analysis = state["analysis"]

        if analysis["requires_current_info"]:
            state["search_method"] = "web"
        elif analysis["type"] == "multi_hop":
            state["search_method"] = "graph"
        elif len(analysis["entities"]) > 3:
            state["search_method"] = "multi"
        else:
            state["search_method"] = "vector"

        return state
```

### 3.5 Chunking Strategies

**Técnicas Avanzadas**:

1. **Semantic Chunking**: Dividir por significado, no por tamaño
2. **Recursive Chunking**: Jerarquía de chunks
3. **Document-Aware Chunking**: Respeta estructura (headers, secciones)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Semantic-aware splitter
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Con metadata preservation
chunks = splitter.create_documents(
    texts=[document.page_content],
    metadatas=[{
        "source": document.metadata["source"],
        "page": document.metadata["page"],
        "section": extract_section(document)
    }]
)
```

### 3.6 Embeddings Optimization

**Modelos de Embedding (2025)**:

```python
from sentence_transformers import SentenceTransformer

# Top embeddings models
models = {
    "general": "sentence-transformers/all-mpnet-base-v2",
    "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "code": "microsoft/codebert-base",
    "domain_specific": "fine-tuned-domain-model"
}

# OpenAI embeddings
from openai import OpenAI
client = OpenAI()

embeddings = client.embeddings.create(
    model="text-embedding-3-large",
    input=text,
    dimensions=3072  # Configurable: 256, 1024, 3072
)
```

**Fine-tuning Embeddings**:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data
train_examples = [
    InputExample(texts=['query 1', 'relevant doc'], label=1.0),
    InputExample(texts=['query 1', 'irrelevant doc'], label=0.0),
]

# Load model
model = SentenceTransformer('base-model')

# Create dataloader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss
train_loss = losses.CosineSimilarityLoss(model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)
```

### 3.7 Vector Databases Comparison

| Database | Strengths | Best For |
|----------|-----------|----------|
| **Pinecone** | Fully managed, scalable | Production at scale |
| **Weaviate** | GraphQL, multimodal | Complex queries |
| **Qdrant** | Fast, Rust-based | High performance |
| **Chroma** | Simple, embeddable | Prototyping |
| **Milvus** | Open source, distributed | Self-hosted scale |

### 3.8 Evaluación de RAG (RAGAS)

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision,
)

# Prepare evaluation data
data = {
    "question": [questions],
    "answer": [generated_answers],
    "contexts": [retrieved_contexts],
    "ground_truths": [reference_answers]
}

# Evaluate
result = evaluate(
    dataset=data,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_recall,
        context_precision,
    ],
)

print(result)
```

## Ejercicios Prácticos

1. **Hybrid RAG**: Implementar sistema con vector + keyword search
2. **GraphRAG**: Construir knowledge graph y query system
3. **Agentic RAG**: Crear agente que decide estrategia de retrieval
4. **RAG Evaluation**: Evaluar sistema con RAGAS

## Recursos

- [LangChain RAG Docs](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)

[← Módulo Anterior](../modulo02_ia_agentica/) | [Siguiente Módulo →](../modulo04_ia_multimodal/)
