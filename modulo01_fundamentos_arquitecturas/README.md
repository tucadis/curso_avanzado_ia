# Módulo 1: Fundamentos Avanzados y Arquitecturas de Nueva Generación

## Descripción General

Este módulo profundiza en las arquitecturas de modelos de lenguaje de última generación, incluyendo GPT-5, Claude Sonnet 4.5, y Gemini 2.5 Pro. Aprenderás cómo funcionan estos modelos, sus diferencias arquitectónicas, y cómo seleccionar el modelo adecuado para cada caso de uso.

## Objetivos de Aprendizaje

Al finalizar este módulo serás capaz de:

- Comprender la evolución de los LLMs desde transformers hasta arquitecturas post-transformers
- Analizar las características únicas de GPT-5, Claude Sonnet 4.5 y Gemini 2.5 Pro
- Evaluar modelos según benchmarks y casos de uso específicos
- Trabajar con Small Language Models (SLMs) para tareas especializadas
- Aplicar técnicas de cuantización y optimización de modelos

## Contenido

### 1.1 Evolución de los LLMs: De GPT-3 a la Era Post-Transformers

#### Historia y Cronología

**Primera Generación (2018-2020)**
- GPT-1 (117M parámetros) - 2018
- BERT (340M parámetros) - 2018
- GPT-2 (1.5B parámetros) - 2019

**Segunda Generación (2020-2022)**
- GPT-3 (175B parámetros) - 2020
- PaLM (540B parámetros) - 2022
- Chinchilla (70B parámetros) - 2022

**Tercera Generación (2022-2024)**
- GPT-4 (parámetros no revelados) - 2023
- Claude 2 - 2023
- Gemini 1.5 Pro - 2024

**Cuarta Generación (2024-2025)**
- GPT-5 - 2025
- Claude Sonnet 4.5 - 2025
- Gemini 2.5 Pro - 2025

#### Arquitectura Transformer: Base Fundamental

La arquitectura Transformer, introducida en "Attention is All You Need" (Vaswani et al., 2017), revolucionó el procesamiento del lenguaje natural con estos componentes clave:

**Mecanismo de Atención Multi-Head**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Componentes:
- **Q (Query)**: Representación de lo que buscamos
- **K (Key)**: Representación de dónde buscar
- **V (Value)**: Contenido real a extraer
- **d_k**: Dimensionalidad de las keys (para escalado)

**Ventajas de Transformers**
- Paralelización eficiente del entrenamiento
- Captura de dependencias a largo plazo
- Escalabilidad a billones de parámetros

**Limitaciones**
- Complejidad O(n²) en atención
- Alto consumo de memoria
- Costos de inferencia significativos

#### Post-Transformers: Nuevas Arquitecturas

**Mamba (State Space Models)**
- Complejidad lineal O(n) vs O(n²)
- Mejor eficiencia en secuencias largas
- Emergente en 2024-2025

**RetNet (Retentive Networks)**
- Entrenamiento paralelo + inferencia recurrente
- Reducción de costos computacionales
- Mantiene capacidades de largo contexto

**RWKV (Receptance Weighted Key Value)**
- Combina RNNs y Transformers
- Complejidad lineal
- Escalable y eficiente

### 1.2 Arquitecturas de Última Generación (2025)

#### GPT-5: Arquitectura Unificada Multimodal

**Características Principales**

1. **Procesamiento Multimodal Nativo**
   - Entrada/salida simultánea: texto, código, imágenes, audio, video
   - Tokenización unificada para todas las modalidades
   - Contexto compartido entre modalidades

2. **Arquitectura Mejorada**
   - Mixture of Experts (MoE) avanzado
   - Sparse attention patterns
   - Contexto extendido: 1M+ tokens

3. **Capacidades de Razonamiento**
   - Chain-of-Thought integrado
   - Test-time compute scaling
   - 94.6% en AIME 2025 (matemáticas universitarias)
   - 88.4% en GPQA Diamond (ciencia doctoral)

4. **Especificaciones Técnicas**
   - Parámetros: Estimado >1 Trillón (arquitectura MoE)
   - Contexto: 1M+ tokens
   - Velocidad: Optimizada vs GPT-4
   - Multimodalidad: Nativa end-to-end

**Casos de Uso Óptimos**
- Investigación científica y académica
- Desarrollo de software complejo
- Análisis multimodal (documentos con imágenes, videos)
- Tareas que requieren razonamiento profundo

#### Claude Sonnet 4.5: El Rey del Código

**Características Principales**

1. **Optimización para Coding**
   - 77.2% en tareas de coding agéntico (SWE-bench)
   - Enfoque en más de 30 horas de trabajo continuo
   - Mejor modelo para desarrollo de software

2. **Arquitectura Constitutional AI**
   - Alineación de valores integrada
   - Menor tasa de alucinaciones
   - Mayor confiabilidad en tareas críticas

3. **Contexto y Rendimiento**
   - Contexto: 200K tokens
   - Velocidad de procesamiento superior
   - Balance óptimo precio/rendimiento

4. **Pricing**
   - Input: $3 por millón de tokens
   - Output: $15 por millón de tokens
   - Competitivo para uso intensivo

**Casos de Uso Óptimos**
- Desarrollo de software y code generation
- Code review y refactoring
- Debugging y optimización
- Tareas que requieren precisión y seguimiento de instrucciones

#### Gemini 2.5 Pro: Pensamiento y Ecosistema

**Características Principales**

1. **Thinking Model**
   - Razonamiento paso a paso transparente
   - Explicaciones detalladas del proceso
   - Mayor precisión en tareas complejas

2. **Integración con Google Ecosystem**
   - Acceso directo a Gmail, Drive, Calendar
   - Búsqueda en tiempo real
   - Multimodalidad con Google services

3. **Capacidades Multimodales**
   - Procesamiento de texto, imágenes, audio, video
   - Generación de código optimizado
   - Análisis de datos complejos

4. **Especificaciones**
   - Contexto: 1M+ tokens (Gemini 2.5)
   - Multimodalidad nativa
   - Velocidad competitiva

**Casos de Uso Óptimos**
- Usuarios del ecosistema Google
- Análisis de grandes volúmenes de información
- Tareas que requieren búsqueda en tiempo real
- Aplicaciones empresariales Google Workspace

### 1.3 Small Language Models (SLMs): Eficiencia y Especialización

#### ¿Por qué SLMs?

En 2025, la tendencia se mueve hacia modelos más pequeños y especializados:

**Ventajas de los SLMs**
- **Menor costo**: Reducción de 10-100x en costos de inferencia
- **Menor latencia**: Respuestas en milisegundos
- **Edge deployment**: Ejecutables en dispositivos locales
- **Privacidad**: Datos permanecen locales
- **Sostenibilidad**: Menor consumo energético

#### SLMs Destacados (2025)

**Phi-4 (Microsoft)**
- 14B parámetros
- Rendimiento comparable a modelos 10x más grandes
- Optimizado para razonamiento matemático y código
- Ejecutable en hardware consumer

**Gemma 2 (Google)**
- 9B y 27B variantes
- Open source bajo licencia permisiva
- Alto rendimiento en benchmarks
- Optimizado para fine-tuning

**Llama 4 Small (Meta)**
- 8B parámetros
- Multilingüe (100+ idiomas)
- Fine-tuning eficiente
- Amplia adopción comunitaria

**Mistral 7B v0.3**
- 7B parámetros
- Mejor relación rendimiento/tamaño
- Sliding window attention
- Excelente para tareas específicas

#### Cuándo Usar SLMs vs LLMs

| Aspecto | SLMs | LLMs |
|---------|------|------|
| **Costo** | Bajo ($0.1-0.5/1M tokens) | Alto ($3-60/1M tokens) |
| **Latencia** | 50-200ms | 500-2000ms |
| **Casos de Uso** | Tareas específicas | Tareas generales complejas |
| **Deployment** | Edge, on-premise | Cloud principalmente |
| **Fine-tuning** | Fácil y económico | Costoso |
| **Razonamiento** | Limitado | Avanzado |

### 1.4 Técnicas de Compresión y Cuantización Avanzadas

#### Cuantización

La cuantización reduce la precisión numérica de los pesos del modelo para disminuir tamaño y acelerar inferencia.

**Niveles de Cuantización**

**FP32 (Full Precision)**
- 32 bits por parámetro
- Máxima precisión
- Uso: Entrenamiento

**FP16 (Half Precision)**
- 16 bits por parámetro
- 2x reducción de memoria
- Minimal pérdida de precisión
- Uso: Inferencia estándar

**INT8 (8-bit Integer)**
- 8 bits por parámetro
- 4x reducción de memoria
- Ligera degradación de rendimiento
- Uso: Producción optimizada

**INT4 (4-bit Integer)**
- 4 bits por parámetro
- 8x reducción de memoria
- Mayor degradación
- Uso: Edge devices

**GGUF (GPT-Generated Unified Format)**
- Formato optimizado para modelos cuantizados
- Soporte de quantización mixta
- Amplio uso con Llama.cpp y Ollama

#### Técnicas Avanzadas

**GPTQ (GPT Quantization)**
```python
from transformers import AutoModelForCausalLM, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**AWQ (Activation-aware Weight Quantization)**
- Protege pesos críticos basándose en activaciones
- Mejor preservación de rendimiento
- Ideal para modelos grandes

**GGML/GGUF**
```bash
# Cuantizar modelo a GGUF
python convert.py model_name --outfile model.gguf
./quantize model.gguf model-q4_k_m.gguf q4_k_m
```

### 1.5 Análisis Comparativo de Modelos: Benchmarks y Casos de Uso

#### Benchmarks Principales

**Coding**
- **HumanEval**: Evaluación de generación de código Python
- **MBPP**: Python programming problems
- **SWE-bench**: Tareas de ingeniería de software real

**Razonamiento**
- **MMLU**: Massive Multitask Language Understanding
- **BBH**: Big Bench Hard
- **GPQA**: Graduate-level science questions

**Matemáticas**
- **GSM8K**: Grade school math
- **MATH**: Competition-level mathematics
- **AIME**: American Invitational Mathematics Examination

#### Tabla Comparativa (2025)

| Modelo | MMLU | HumanEval | MATH | Contexto | Precio ($/1M in) |
|--------|------|-----------|------|----------|------------------|
| **GPT-5** | 92.5% | 92.0% | 94.6% | 1M+ | $30-60 |
| **Claude Sonnet 4.5** | 91.2% | 93.7% | 89.5% | 200K | $3 |
| **Gemini 2.5 Pro** | 90.8% | 90.1% | 91.2% | 1M+ | $2.5-7 |
| **Llama 4 405B** | 88.6% | 84.2% | 73.8% | 128K | Open Source |
| **Mistral Large 3** | 86.5% | 82.1% | 69.5% | 128K | $2 |

#### Matriz de Decisión

**Elige GPT-5 si:**
- Necesitas máximo rendimiento en razonamiento
- Trabajas con tareas multimodales complejas
- El presupuesto no es limitante
- Requieres las últimas capacidades de IA

**Elige Claude Sonnet 4.5 si:**
- Enfoque en desarrollo de software
- Necesitas alta precisión y confiabilidad
- Buscas mejor relación calidad-precio
- Tareas que requieren seguimiento estricto de instrucciones

**Elige Gemini 2.5 Pro si:**
- Usas ecosistema Google
- Necesitas búsqueda en tiempo real
- Trabajas con grandes contextos
- Requieres integración con Google Workspace

**Elige Modelos Open Source si:**
- Necesitas control total del modelo
- Privacidad de datos es crítica
- Deployment on-premise
- Presupuesto muy limitado
- Fine-tuning extensivo

## Ejercicios Prácticos

### Ejercicio 1: Comparación de Modelos
Implementar llamadas a GPT-5, Claude Sonnet 4.5 y Gemini 2.5 Pro con la misma tarea y comparar resultados.

### Ejercicio 2: Benchmarking Personalizado
Crear un benchmark personalizado para tu caso de uso específico.

### Ejercicio 3: Cuantización de Modelos
Cuantizar un modelo open source y medir el trade-off rendimiento/tamaño.

### Ejercicio 4: Selección de Modelo
Dado un caso de uso empresarial, justificar la selección del modelo óptimo.

## Recursos Adicionales

### Papers Fundamentales
- "Attention is All You Need" - Vaswani et al., 2017
- "Language Models are Few-Shot Learners" - Brown et al., 2020
- "Constitutional AI: Harmlessness from AI Feedback" - Bai et al., 2022
- "Mamba: Linear-Time Sequence Modeling" - Gu & Dao, 2024

### Documentación Oficial
- [OpenAI GPT-5 Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com)
- [Google Gemini Documentation](https://ai.google.dev/docs)

### Herramientas
- [Hugging Face Model Hub](https://huggingface.co/models)
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Ollama](https://ollama.ai) - Run models locally

## Siguientes Pasos

Una vez completado este módulo, estarás preparado para:
- [Módulo 2: Inteligencia Artificial Agéntica →](../modulo02_ia_agentica/)
- Implementar soluciones con los modelos más avanzados
- Evaluar y seleccionar modelos para casos de uso específicos
- Optimizar modelos para producción

---

[← Volver al Índice Principal](../README.md) | [Siguiente Módulo →](../modulo02_ia_agentica/)
