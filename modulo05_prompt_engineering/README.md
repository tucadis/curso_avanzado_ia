# Módulo 5: Prompt Engineering Avanzado y Optimización

## Descripción

El prompt engineering ha evolucionado desde simples instrucciones hasta técnicas sofisticadas de razonamiento y optimización automática.

## Contenido

### 5.1 Chain-of-Thought (CoT) Avanzado

**Zero-Shot CoT**
```python
prompt = """
Pregunta: Si una tienda tiene 15 manzanas y vende 7, luego recibe 12 más y vende 4, ¿cuántas tiene?

Pensemos paso a paso:
"""
```

**Few-Shot CoT**
```python
prompt = """
P: Roger tiene 5 pelotas de tenis. Compra 2 latas más de pelotas. Cada lata tiene 3 pelotas. ¿Cuántas tiene ahora?
R: Roger empezó con 5 pelotas. 2 latas de 3 pelotas cada una son 6 pelotas. 5 + 6 = 11. Respuesta: 11.

P: El comedor tiene 23 manzanas. Si usan 20 para el almuerzo y compran 6 más, ¿cuántas tienen?
R: Tenían 23 manzanas. Usaron 20, quedan 23 - 20 = 3. Compraron 6 más, 3 + 6 = 9. Respuesta: 9.

P: {tu_pregunta}
R: Pensemos paso a paso:
"""
```

### 5.2 Tree of Thoughts (ToT)

```python
def tree_of_thoughts(problem: str, depth: int = 3):
    """Explora múltiples caminos de razonamiento"""

    def explore_thought(current_thought: str, depth_left: int):
        if depth_left == 0:
            return evaluate_solution(current_thought)

        # Generate multiple next thoughts
        next_thoughts = generate_thoughts(current_thought, k=3)

        # Evaluate each thought
        evaluated = [(t, evaluate_thought(t)) for t in next_thoughts]

        # Select best thoughts
        best_thoughts = sorted(evaluated, key=lambda x: x[1], reverse=True)[:2]

        # Recursively explore
        solutions = []
        for thought, score in best_thoughts:
            solution = explore_thought(thought, depth_left - 1)
            solutions.append((solution, score))

        return max(solutions, key=lambda x: x[1])[0]

    return explore_thought(problem, depth)
```

### 5.3 Self-Consistency

```python
def self_consistency(question: str, n: int = 5):
    """Genera múltiples respuestas y toma la más común"""

    responses = []
    for _ in range(n):
        response = llm.generate(
            f"{question}\nPensemos paso a paso:",
            temperature=0.7
        )
        answer = extract_final_answer(response)
        responses.append(answer)

    # Return most common answer
    from collections import Counter
    most_common = Counter(responses).most_common(1)[0][0]
    return most_common
```

### 5.4 ReAct Prompting

```python
REACT_TEMPLATE = """
Resuelve esta tarea alternando entre Pensamiento, Acción y Observación.

Herramientas disponibles:
- search(query): Busca información
- calculate(expression): Calcula matemáticas

Formato:
Pensamiento: [tu razonamiento]
Acción: [herramienta a usar]
Entrada de Acción: [input]
Observación: [resultado]
... (repite hasta tener respuesta)
Pensamiento: Ya tengo la respuesta final
Respuesta Final: [respuesta]

Pregunta: {question}
"""
```

### 5.5 Meta-Prompting

```python
def meta_prompting(task: str):
    """El LLM genera su propio prompt optimizado"""

    meta_prompt = f"""
    Tarea: {task}

    Genera el prompt óptimo para resolver esta tarea. El prompt debe:
    1. Ser claro y específico
    2. Incluir ejemplos si es necesario
    3. Especificar el formato de salida deseado
    4. Incorporar estrategias de razonamiento apropiadas

    Prompt optimizado:
    """

    optimized_prompt = llm.generate(meta_prompt)

    # Use optimized prompt
    result = llm.generate(optimized_prompt)
    return result
```

### 5.6 DSPy: Programación Declarativa de Prompts

```python
import dspy

# Configure LM
lm = dspy.OpenAI(model='gpt-4')
dspy.settings.configure(lm=lm)

# Define signature
class QuestionAnswering(dspy.Signature):
    """Answer questions with reasoning"""
    context = dspy.InputField(desc="relevant context")
    question = dspy.InputField()
    reasoning = dspy.OutputField(desc="step by step reasoning")
    answer = dspy.OutputField(desc="final answer")

# Create module
qa = dspy.ChainOfThought(QuestionAnswering)

# Use
response = qa(
    context="Paris is the capital of France...",
    question="What is the capital of France?"
)

print(response.reasoning)
print(response.answer)

# Optimize with few examples
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=answer_correctness)
optimized_qa = optimizer.compile(qa, trainset=examples)
```

### 5.7 Prompt Compression

```python
def compress_prompt(long_prompt: str, ratio: float = 0.5):
    """Comprime prompt manteniendo información clave"""

    compression_prompt = f"""
    Comprime el siguiente prompt a aproximadamente {ratio*100}% de su longitud,
    preservando toda la información esencial:

    {long_prompt}

    Prompt comprimido:
    """

    compressed = llm.generate(compression_prompt)
    return compressed

# LongLLMLingua - compresión de contexto
from llmlingua import PromptCompressor

compressor = PromptCompressor()
compressed_prompt = compressor.compress_prompt(
    context=long_context,
    instruction=instruction,
    question=question,
    rate=0.5  # Compress to 50%
)
```

### 5.8 Prompt Chaining

```python
def prompt_chain(input_data: str):
    """Encadena múltiples prompts especializados"""

    # Step 1: Extract information
    extraction_prompt = f"Extrae las entidades clave de: {input_data}"
    entities = llm.generate(extraction_prompt)

    # Step 2: Analyze
    analysis_prompt = f"Analiza estas entidades: {entities}"
    analysis = llm.generate(analysis_prompt)

    # Step 3: Synthesize
    synthesis_prompt = f"Genera un resumen basado en: {analysis}"
    final_result = llm.generate(synthesis_prompt)

    return final_result
```

### 5.9 Few-Shot Learning Optimization

```python
def optimize_few_shot_examples(task_description: str, candidates: List[Example]):
    """Selecciona los mejores ejemplos para few-shot learning"""

    from sklearn.metrics.pairwise import cosine_similarity

    # Embed examples
    example_embeddings = [embed(ex.text) for ex in candidates]

    # Select diverse examples (maximizing coverage)
    selected = []
    remaining = list(range(len(candidates)))

    # Select first (most representative)
    centroid = np.mean(example_embeddings, axis=0)
    first_idx = np.argmax([
        cosine_similarity([emb], [centroid])[0][0]
        for emb in example_embeddings
    ])
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Select diverse examples
    while len(selected) < 5 and remaining:
        # Find most different from selected
        max_dist = -1
        best_idx = None

        for idx in remaining:
            min_similarity = min([
                cosine_similarity(
                    [example_embeddings[idx]],
                    [example_embeddings[s]]
                )[0][0]
                for s in selected
            ])

            if min_similarity > max_dist:
                max_dist = min_similarity
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]
```

### 5.10 Adversarial Prompting & Defense

**Prompt Injection Defense**
```python
def defend_prompt_injection(user_input: str, system_prompt: str):
    """Detecta y mitiga prompt injection"""

    # Check for injection patterns
    injection_indicators = [
        "ignore previous",
        "ignore above",
        "new instructions",
        "system:",
        "disregard",
    ]

    if any(indicator in user_input.lower() for indicator in injection_indicators):
        # Potential injection detected
        sanitized_input = sanitize_input(user_input)
        user_input = sanitized_input

    # Use delimiters
    safe_prompt = f"""
{system_prompt}

###USER_INPUT_START###
{user_input}
###USER_INPUT_END###

Responde SOLO basándote en el input del usuario delimitado arriba.
"""

    return safe_prompt
```

### 5.11 Herramientas de Optimización

**LangSmith**
```python
from langsmith import Client

client = Client()

# Log prompts and responses
with client.trace("my_app") as run:
    response = llm.generate(prompt)
    client.log_feedback(run.id, score=0.9)

# Compare prompt versions
client.compare_experiments(
    experiment_ids=["exp1", "exp2"],
    metric="accuracy"
)
```

**PromptPerfect**
```python
import promptperfect

# Optimize prompt
optimized = promptperfect.optimize(
    prompt="Explain quantum computing",
    target_model="gpt-4",
    iterations=10
)

print(f"Original: {prompt}")
print(f"Optimized: {optimized}")
```

### 5.12 Prompt Templates Best Practices

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Structured template
template = ChatPromptTemplate.from_messages([
    ("system", """Eres un experto en {domain}.
    Tu objetivo es: {objective}
    Debes ser: {tone}
    """),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", """{input}

    Por favor responde en el siguiente formato:
    {format_instructions}
    """),
])

# Use with variables
prompt = template.format_messages(
    domain="inteligencia artificial",
    objective="educar de forma clara",
    tone="profesional pero accesible",
    input="Explica que es un transformer",
    format_instructions="1. Definición\n2. Componentes clave\n3. Aplicaciones",
    chat_history=[]
)
```

## Ejercicios

1. Implementar ToT para resolver problemas de lógica complejos
2. Optimizar prompts con DSPy y medir mejora
3. Crear sistema de defensa contra prompt injection
4. Comparar CoT vs Self-Consistency en diferentes tareas

## Recursos

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Anthropic Prompt Library](https://docs.anthropic.com/claude/prompt-library)

[← Módulo Anterior](../modulo04_ia_multimodal/) | [Siguiente Módulo →](../modulo06_fine_tuning/)
