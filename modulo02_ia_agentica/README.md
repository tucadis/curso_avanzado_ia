# Módulo 2: Inteligencia Artificial Agéntica (Agentic AI)

## Descripción

La IA Agéntica representa un cambio de paradigma en cómo construimos sistemas de IA. En lugar de modelos que simplemente responden a prompts, los agentes pueden planificar, ejecutar acciones, usar herramientas, y trabajar de forma autónoma hacia objetivos complejos.

## Objetivos de Aprendizaje

- Comprender los fundamentos de agentes autónomos de IA
- Implementar agentes usando ReAct, AutoGPT, y frameworks modernos
- Diseñar sistemas multi-agente con coordinación
- Aplicar IA agéntica a casos de uso empresariales
- Evaluar y optimizar el rendimiento de agentes

## Contenido

### 2.1 Fundamentos de IA Agéntica

#### ¿Qué es un Agente de IA?

Un agente de IA es un sistema que:
1. **Percibe** su entorno
2. **Razona** sobre objetivos y acciones
3. **Actúa** para lograr objetivos
4. **Aprende** de resultados

**Diferencias clave vs LLMs tradicionales:**

| Aspecto | LLM Tradicional | Agente de IA |
|---------|----------------|--------------|
| **Interacción** | Única (prompt → respuesta) | Múltiple e iterativa |
| **Autonomía** | Ninguna | Alta |
| **Herramientas** | No usa herramientas | Puede usar APIs, bases de datos, etc. |
| **Planificación** | No planifica | Crea y ejecuta planes |
| **Memoria** | Solo contexto | Memoria a corto y largo plazo |

#### Componentes de un Agente

```
┌─────────────────────────────────────────┐
│           AGENTE DE IA                   │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────┐    ┌──────────────┐   │
│  │   Memoria   │◄──►│  Razonamiento │   │
│  └─────────────┘    └──────────────┘   │
│         ▲                   │           │
│         │                   ▼           │
│  ┌─────────────┐    ┌──────────────┐   │
│  │ Herramientas│◄──►│  Planificador │   │
│  └─────────────┘    └──────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

1. **LLM Core**: Motor de razonamiento
2. **Memoria**:
   - Corto plazo: Contexto actual
   - Largo plazo: Base de conocimiento persistente
3. **Herramientas**: Funciones que el agente puede llamar
4. **Planificador**: Estrategia para lograr objetivos
5. **Executor**: Ejecuta acciones

### 2.2 Arquitecturas de Agentes

#### ReAct (Reasoning + Acting)

ReAct alterna entre razonamiento y acción:

```
Pensamiento: Necesito buscar información sobre Python
Acción: search("Python programming")
Observación: Python es un lenguaje...
Pensamiento: Ahora entiendo, puedo responder
Acción: respond(answer)
```

**Implementación conceptual:**
```python
def react_loop(question, max_iterations=5):
    for i in range(max_iterations):
        # Reasoning
        thought = llm(f"Pensamiento sobre: {question}")

        # Acting
        if should_use_tool(thought):
            action = extract_action(thought)
            observation = execute_tool(action)
            question = f"{question}\nObservación: {observation}"
        else:
            return extract_answer(thought)
```

#### Plan-and-Execute

1. **Planificación**: Crear plan completo primero
2. **Ejecución**: Ejecutar pasos secuencialmente
3. **Re-planificación**: Ajustar si es necesario

```python
# Ejemplo de plan
plan = [
    "1. Buscar datos de ventas del Q4 2024",
    "2. Calcular promedio mensual",
    "3. Comparar con Q4 2023",
    "4. Generar visualización",
    "5. Crear reporte ejecutivo"
]
```

#### ReWOO (Reasoning WithOut Observation)

Optimización de ReAct que separa planificación de ejecución:
- Planifica todas las acciones primero
- Ejecuta en paralelo cuando sea posible
- Más eficiente que ReAct

### 2.3 Frameworks y Herramientas

#### LangGraph

Framework de Anthropic para construir agentes con grafos:

```python
from langgraph.graph import StateGraph, END

# Definir estado
class AgentState(TypedDict):
    messages: list
    next_action: str

# Crear grafo
workflow = StateGraph(AgentState)

# Añadir nodos
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Añadir edges
workflow.add_edge("agent", "tools")
workflow.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

# Compilar
app = workflow.compile()
```

**Ventajas de LangGraph:**
- Control fino del flujo
- Ciclos y condicionales
- Persistencia de estado
- Debugging visual

#### CrewAI

Framework para equipos de agentes especializados:

```python
from crewai import Agent, Task, Crew

# Definir agentes
researcher = Agent(
    role='Investigador',
    goal='Encontrar información precisa',
    backstory='Experto en investigación...',
    tools=[search_tool, scraper_tool]
)

writer = Agent(
    role='Escritor',
    goal='Crear contenido de calidad',
    backstory='Escritor profesional...',
    tools=[writing_tool]
)

# Definir tareas
research_task = Task(
    description='Investigar IA generativa 2025',
    agent=researcher
)

writing_task = Task(
    description='Escribir artículo basado en investigación',
    agent=writer
)

# Crear crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

# Ejecutar
result = crew.kickoff()
```

#### AutoGen (Microsoft)

Framework para conversaciones multi-agente:

```python
from autogen import AssistantAgent, UserProxyAgent

# Agente asistente
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

# Agente proxy (ejecuta código)
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"}
)

# Iniciar conversación
user_proxy.initiate_chat(
    assistant,
    message="Analiza estas ventas y crea un gráfico"
)
```

### 2.4 Herramientas (Tools) para Agentes

Los agentes se vuelven poderosos cuando pueden usar herramientas:

#### Tipos de Herramientas

1. **Búsqueda y Recuperación**
   - Web search (Google, Bing)
   - Database queries
   - Vector search

2. **Ejecución de Código**
   - Python REPL
   - Shell commands
   - Jupyter notebooks

3. **APIs Externas**
   - Weather APIs
   - Financial data
   - Company databases

4. **Procesamiento de Archivos**
   - PDF readers
   - Excel processors
   - Image analyzers

#### Definición de Herramientas

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Útil para cálculos matemáticos. Input: expresión matemática."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error en cálculo"

@tool
def web_search(query: str) -> str:
    """Busca información en internet. Input: consulta de búsqueda."""
    # Implementación de búsqueda
    results = search_api(query)
    return results

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Envía un email. Input: destinatario, asunto, cuerpo."""
    # Implementación de envío
    send_email_api(to, subject, body)
    return "Email enviado exitosamente"
```

### 2.5 Sistemas Multi-Agente

#### Patrones de Coordinación

**1. Jerárquico**
```
    Manager Agent
    ┌────┴────┐
Worker1   Worker2   Worker3
```

**2. Secuencial**
```
Agent1 → Agent2 → Agent3 → Result
```

**3. Paralelo con Agregación**
```
       ┌→ Agent1 ─┐
Task ─→┼→ Agent2 ─┼→ Aggregator → Result
       └→ Agent3 ─┘
```

**4. Colaborativo**
```
Agent1 ↔ Agent2
  ↕        ↕
Agent3 ↔ Agent4
```

#### Ejemplo: Equipo de Investigación

```python
# Equipo especializado
team = {
    "researcher": Agent(
        role="Investigador",
        tools=[search, database],
        specialization="Recopilar información"
    ),
    "analyst": Agent(
        role="Analista",
        tools=[calculator, plotter],
        specialization="Analizar datos"
    ),
    "writer": Agent(
        role="Escritor",
        tools=[template, formatter],
        specialization="Crear reportes"
    ),
    "reviewer": Agent(
        role="Revisor",
        tools=[checker, validator],
        specialization="Control de calidad"
    )
}

# Flujo de trabajo
def research_workflow(topic):
    # 1. Investigar
    data = team["researcher"].execute(f"Investiga {topic}")

    # 2. Analizar
    analysis = team["analyst"].execute(f"Analiza: {data}")

    # 3. Escribir
    draft = team["writer"].execute(f"Escribe reporte: {analysis}")

    # 4. Revisar
    final = team["reviewer"].execute(f"Revisa: {draft}")

    return final
```

### 2.6 Memoria en Agentes

#### Tipos de Memoria

**1. Memoria de Conversación (Short-term)**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "Hola, soy Juan"},
    {"output": "Hola Juan, ¿en qué puedo ayudarte?"}
)
```

**2. Memoria Vectorial (Long-term)**
```python
from langchain.memory import VectorStoreRetrieverMemory

memory = VectorStoreRetrieverMemory(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5})
)
```

**3. Memoria de Entidades**
```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=llm)
# Extrae y recuerda información sobre entidades (personas, lugares, etc.)
```

**4. Memoria Sumaria**
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
# Resume conversaciones largas automáticamente
```

### 2.7 Evaluación de Agentes

#### Métricas Clave

1. **Task Success Rate**: % de tareas completadas exitosamente
2. **Efficiency**: Número de pasos para completar tarea
3. **Cost**: Tokens/llamadas usadas
4. **Latency**: Tiempo hasta completar
5. **Tool Usage Accuracy**: Uso correcto de herramientas

#### Benchmarks

- **SWE-bench**: Tareas de ingeniería de software
- **WebArena**: Navegación y tareas web
- **GAIA**: General AI Assistants benchmark
- **AgentBench**: Benchmark multidominio

## Ejercicios Prácticos

### Ejercicio 1: Agente ReAct Simple
Implementar un agente ReAct con búsqueda web y calculadora.

### Ejercicio 2: Sistema Multi-Agente
Crear un equipo de agentes para análisis de datos: recolector, analista, visualizador.

### Ejercicio 3: Agente con Memoria
Implementar un asistente personal que recuerde preferencias del usuario.

### Ejercicio 4: Optimización de Agentes
Comparar eficiencia de ReAct vs Plan-and-Execute en tareas complejas.

## Recursos Adicionales

### Papers
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023)
- "Generative Agents: Interactive Simulacra of Human Behavior" (Park et al., 2023)
- "AutoGPT: An Autonomous GPT-4 Experiment"

### Frameworks
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AutoGen](https://github.com/microsoft/autogen)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)

### Herramientas
- [LangSmith](https://smith.langchain.com/) - Debugging de agentes
- [AgentOps](https://www.agentops.ai/) - Observabilidad
- [Traces](https://github.com/langchain-ai/langchain-traces) - Visualización

## Siguientes Pasos

[Módulo 3: RAG Avanzado y Agentic RAG →](../modulo03_rag_avanzado/)

---

[← Módulo Anterior](../modulo01_fundamentos_arquitecturas/) | [Volver al Índice](../README.md)
