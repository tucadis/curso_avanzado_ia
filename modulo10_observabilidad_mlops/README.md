# Módulo 10: Observabilidad, Evaluación y MLOps

## Contenido

### 10.1 LangSmith - Observability

```python
from langsmith import Client
from langchain.callbacks import LangSmithCallbackHandler

client = Client()

# Trace execution
with client.trace("my_app", tags=["production"]) as run:
    response = chain.invoke(input_data)

    # Log feedback
    client.log_feedback(
        run.id,
        key="user_score",
        score=0.9
    )

# Search traces
runs = client.list_runs(
    project_name="my_app",
    filter="eq(metadata.user, 'john')"
)

# Compare experiments
client.compare_runs(
    baseline_run_id="run1",
    comparison_run_ids=["run2", "run3"]
)
```

### 10.2 Weights & Biases

```python
import wandb
from wandb.integration.langchain import WandbTracer

# Initialize
wandb.init(project="llm-project")

# Log metrics
wandb.log({
    "latency": 234,
    "tokens_used": 150,
    "cost": 0.002,
    "user_rating": 4.5
})

# Log prompts and outputs
wandb.log({
    "prompt": prompt,
    "output": output,
    "model": "gpt-4"
})

# Log tables
table = wandb.Table(
    columns=["prompt", "output", "score"],
    data=[[p, o, s] for p, o, s in zip(prompts, outputs, scores)]
)
wandb.log({"predictions": table})
```

### 10.3 Métricas de Evaluación

**Automated Metrics**
```python
from evaluate import load

# BLEU (translation)
bleu = load("bleu")
score = bleu.compute(predictions=preds, references=refs)

# ROUGE (summarization)
rouge = load("rouge")
score = rouge.compute(predictions=summaries, references=references)

# BERTScore (semantic similarity)
bertscore = load("bertscore")
score = bertscore.compute(
    predictions=preds,
    references=refs,
    model_type="microsoft/deberta-xlarge-mnli"
)
```

**LLM-as-Judge**
```python
def llm_evaluate(question: str, answer: str, reference: str) -> dict:
    eval_prompt = f"""
    Evalúa la siguiente respuesta:

    Pregunta: {question}
    Respuesta: {answer}
    Referencia: {reference}

    Califica de 1-5:
    - Precisión (accuracy)
    - Relevancia (relevance)
    - Completitud (completeness)

    Formato JSON:
    """

    evaluation = llm.generate(eval_prompt)
    return json.loads(evaluation)
```

### 10.4 A/B Testing

```python
class ABTest:
    def __init__(self):
        self.variants = {
            "A": model_a,
            "B": model_b
        }
        self.results = {"A": [], "B": []}

    def assign_variant(self, user_id: str) -> str:
        # Consistent hashing
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return "A" if hash_val % 2 == 0 else "B"

    def get_response(self, user_id: str, prompt: str):
        variant = self.assign_variant(user_id)
        model = self.variants[variant]

        response = model.generate(prompt)

        # Log
        self.results[variant].append({
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        })

        return response, variant

    def analyze(self):
        # Compare metrics
        for variant in ["A", "B"]:
            data = self.results[variant]
            print(f"Variant {variant}:")
            print(f"  Count: {len(data)}")
            print(f"  Avg latency: {np.mean([d['latency'] for d in data])}")
```

### 10.5 CI/CD para LLM Apps

**GitHub Actions**
```yaml
name: LLM Pipeline

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run tests
        run: |
          pytest tests/

      - name: Evaluate prompts
        run: |
          python evaluate_prompts.py

      - name: Check performance
        run: |
          python benchmark.py
          if [ $LATENCY > 500 ]; then
            exit 1
          fi

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          kubectl apply -f k8s/deployment.yaml
```

### 10.6 Cost Tracking

```python
class CostTracker:
    PRICING = {
        "gpt-4": {"input": 30/1e6, "output": 60/1e6},
        "gpt-3.5-turbo": {"input": 0.5/1e6, "output": 1.5/1e6},
        "claude-3": {"input": 3/1e6, "output": 15/1e6}
    }

    def __init__(self):
        self.usage = []

    def log_usage(self, model: str, input_tokens: int, output_tokens: int):
        cost = (
            input_tokens * self.PRICING[model]["input"] +
            output_tokens * self.PRICING[model]["output"]
        )

        self.usage.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": datetime.now()
        })

    def get_daily_cost(self) -> float:
        today = datetime.now().date()
        today_usage = [
            u for u in self.usage
            if u["timestamp"].date() == today
        ]
        return sum(u["cost"] for u in today_usage)
```

### 10.7 Monitoring Dashboard

```python
import streamlit as st
import plotly.express as px

st.title("LLM Application Dashboard")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Requests Today", requests_count, delta="+12%")
col2.metric("Avg Latency", f"{avg_latency}ms", delta="-5%")
col3.metric("Cost Today", f"${daily_cost:.2f}", delta="+$2.30")
col4.metric("Success Rate", f"{success_rate}%", delta="+2%")

# Charts
fig = px.line(df, x="timestamp", y="latency", title="Latency Over Time")
st.plotly_chart(fig)

# Errors
st.subheader("Recent Errors")
st.dataframe(errors_df)

# Model comparison
st.subheader("Model Performance")
comparison_df = pd.DataFrame({
    "Model": ["GPT-4", "Claude", "Llama"],
    "Latency": [234, 189, 156],
    "Cost": [0.05, 0.02, 0.001],
    "Quality": [0.95, 0.92, 0.85]
})
st.dataframe(comparison_df)
```

## Ejercicios

1. Configurar LangSmith para tracking completo
2. Implementar A/B testing para comparar modelos
3. Crear dashboard de monitoreo con Streamlit
4. Establecer CI/CD pipeline para app LLM

[← Módulo Anterior](../modulo09_guardrails_seguridad/) | [Siguiente →](../modulo11_casos_uso_empresariales/)
