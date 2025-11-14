# Módulo 12: Sostenibilidad y Eficiencia Energética

## Contenido

### 12.1 Impacto Ambiental de la IA

**Carbon Footprint de Modelos**

| Modelo | Entrenamiento (tCO2e) | Inferencia (1M queries) |
|--------|----------------------|-------------------------|
| GPT-3 175B | ~500 | ~10 |
| GPT-4 | ~2000 (estimado) | ~15-20 |
| Llama 2 70B | ~180 | ~5 |
| Mistral 7B | ~20 | ~1 |

### 12.2 Green AI Principles

**1. Model Selection**
```python
def select_green_model(task_complexity: str, accuracy_threshold: float):
    """Selecciona el modelo más pequeño que cumple requisitos"""

    models = [
        {"name": "gpt-4", "accuracy": 0.95, "carbon": 20},
        {"name": "claude-3", "accuracy": 0.93, "carbon": 15},
        {"name": "llama-70b", "accuracy": 0.88, "carbon": 5},
        {"name": "mistral-7b", "accuracy": 0.82, "carbon": 1}
    ]

    # Filter by accuracy
    candidates = [m for m in models if m["accuracy"] >= accuracy_threshold]

    # Select lowest carbon
    return min(candidates, key=lambda x: x["carbon"])
```

**2. Caching Agresivo**
```python
class GreenCache:
    """Cache with carbon tracking"""

    def __init__(self):
        self.cache = {}
        self.carbon_saved = 0

    def get_or_compute(self, key: str, compute_fn):
        if key in self.cache:
            # Cache hit - no computation needed
            self.carbon_saved += self.estimate_carbon_saved()
            return self.cache[key]

        # Cache miss - must compute
        result = compute_fn()
        self.cache[key] = result
        return result

    def estimate_carbon_saved(self):
        # Estimated CO2 saved per cached response (grams)
        return 0.01  # 10g CO2 per query avoided
```

**3. Batch Processing**
```python
def batch_process_green(requests: List[str], batch_size: int = 32):
    """Process in batches to maximize efficiency"""

    results = []

    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]

        # Single API call for batch
        batch_results = llm.batch_generate(batch)

        results.extend(batch_results)

    return results

# vs individual calls - 32x more efficient!
```

### 12.3 Optimización Energética

**Quantization for Efficiency**
```python
# FP16 uses 50% less energy than FP32
# INT8 uses 75% less energy
# INT4 uses 87.5% less energy

def calculate_energy_savings(model_size_gb: float, quantization: str):
    base_energy = model_size_gb * 10  # Watts per GB (FP32)

    savings = {
        "fp16": 0.5,
        "int8": 0.75,
        "int4": 0.875
    }

    energy_saved = base_energy * savings.get(quantization, 0)

    return {
        "base_energy": base_energy,
        "optimized_energy": base_energy - energy_saved,
        "savings_percentage": savings.get(quantization, 0) * 100
    }
```

**Pruning & Distillation**
```python
from transformers import DistillationTrainer

# Knowledge distillation - smaller model, similar performance
teacher = AutoModelForCausalLM.from_pretrained("gpt-2-xl")  # 1.5B
student = AutoModelForCausalLM.from_pretrained("gpt-2")    # 117M

trainer = DistillationTrainer(
    student_model=student,
    teacher_model=teacher,
    train_dataset=dataset
)

trainer.train()

# Student model: 10x smaller, 90% performance, 90% less energy
```

### 12.4 Edge Computing

```python
# Deploy small models at edge instead of cloud

class EdgeDeployment:
    """Deploy optimized models on edge devices"""

    def __init__(self):
        # Use TFLite or ONNX for edge
        self.model = self.load_edge_model()

    def load_edge_model(self):
        # Quantized, pruned model for mobile/edge
        model = onnx.load("model-int8.onnx")
        return model

    def infer_local(self, input_data):
        # Inference on device - zero network energy!
        return self.model.predict(input_data)

# Benefits:
# - No data transmission energy
# - No datacenter energy
# - Privacy preservation
# - Lower latency
```

### 12.5 Renewable Energy-Aware Scheduling

```python
class GreenScheduler:
    """Schedule compute-intensive tasks when renewable energy is high"""

    def __init__(self):
        self.electricity_api = ElectricityMapsAPI()

    def get_carbon_intensity(self, region: str) -> float:
        """Get current carbon intensity (gCO2/kWh)"""
        return self.electricity_api.get_intensity(region)

    def should_process_now(self, region: str, threshold: int = 200):
        """Check if carbon intensity is low enough"""
        intensity = self.get_carbon_intensity(region)
        return intensity < threshold

    async def schedule_green(self, task, region: str):
        """Wait for low carbon intensity"""
        while not self.should_process_now(region):
            await asyncio.sleep(300)  # Check every 5 minutes

        # Process when green energy is available
        return await task()
```

### 12.6 Carbon Tracking

```python
class CarbonTracker:
    """Track and report carbon emissions"""

    # Carbon intensity per kWh by region (gCO2/kWh)
    GRID_INTENSITY = {
        "us-east": 400,
        "us-west": 300,
        "eu-north": 50,   # High renewable
        "eu-central": 250
    }

    # Model energy consumption (kWh per 1M tokens)
    MODEL_ENERGY = {
        "gpt-4": 10,
        "gpt-3.5": 3,
        "claude-3": 5,
        "llama-70b": 2,
        "mistral-7b": 0.5
    }

    def calculate_emissions(
        self,
        model: str,
        tokens: int,
        region: str
    ) -> dict:
        """Calculate CO2 emissions for inference"""

        # Energy used (kWh)
        energy = (tokens / 1_000_000) * self.MODEL_ENERGY.get(model, 5)

        # Carbon emissions (gCO2)
        carbon = energy * self.GRID_INTENSITY.get(region, 300)

        return {
            "energy_kwh": energy,
            "carbon_gco2": carbon,
            "carbon_equivalent": self.get_equivalent(carbon)
        }

    def get_equivalent(self, carbon_grams: float) -> str:
        """Human-readable equivalents"""
        if carbon_grams < 100:
            return f"{carbon_grams:.1f}g CO2 (charging a smartphone)"
        elif carbon_grams < 1000:
            return f"{carbon_grams/1000:.2f}kg CO2 (boiling a kettle)"
        else:
            km = carbon_grams / 120  # 120g CO2 per km driving
            return f"{carbon_grams/1000:.2f}kg CO2 ({km:.1f}km driving)"

# Usage
tracker = CarbonTracker()
emissions = tracker.calculate_emissions(
    model="gpt-4",
    tokens=5000,
    region="us-east"
)

print(f"Emissions: {emissions['carbon_equivalent']}")
```

### 12.7 Best Practices

**1. Choose Appropriate Model Size**
```python
# Don't use GPT-4 for simple classification
# Good:
result = small_model.classify(text)

# Bad:
result = gpt4.generate(f"Classify: {text}")
```

**2. Implement Smart Caching**
```python
# Cache similar queries
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.cache = []
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def get(self, query: str):
        query_emb = self.encoder.encode(query)

        for cached_query, cached_emb, result in self.cache:
            similarity = cosine_similarity(query_emb, cached_emb)
            if similarity > self.similarity_threshold:
                return result  # Cache hit!

        return None
```

**3. Use Regional Data Centers**
```python
# Choose datacenter with lowest carbon intensity
def select_region(regions: List[str]) -> str:
    intensities = {r: get_carbon_intensity(r) for r in regions}
    return min(intensities, key=intensities.get)
```

### 12.8 Reporting & Compliance

```python
class SustainabilityReport:
    def generate_monthly_report(self):
        report = {
            "total_requests": self.get_request_count(),
            "energy_consumed_kwh": self.get_energy_consumed(),
            "carbon_emissions_kg": self.get_carbon_emissions(),
            "green_energy_percentage": self.get_green_percentage(),
            "efficiency_improvements": self.get_efficiency_gains(),
            "recommendations": self.get_recommendations()
        }

        return report

    def get_recommendations(self):
        return [
            "Increase caching ratio (currently 60%, target 80%)",
            "Migrate 20% of workload to eu-north (high renewable)",
            "Implement request batching for 15% energy savings",
            "Consider model distillation for 30% smaller model"
        ]
```

## Ejercicios

1. Implementar carbon tracker para aplicación LLM
2. Optimizar modelo con quantization y medir ahorro energético
3. Crear scheduler con renewable energy awareness
4. Desarrollar sustainability dashboard

[← Módulo Anterior](../modulo11_casos_uso_empresariales/) | [Siguiente →](../modulo13_etica_legal/)
