# Módulo 6: Fine-Tuning y Personalización de Modelos

## Contenido

### 6.1 Estrategias de Fine-Tuning

**Full Fine-Tuning vs PEFT**

| Método | Parámetros Entrenables | Memoria | Tiempo | Costo |
|--------|----------------------|---------|--------|-------|
| Full FT | 100% | Alto | Alto | Alto |
| LoRA | ~0.1% | Bajo | Bajo | Bajo |
| QLoRA | ~0.1% | Muy Bajo | Bajo | Muy Bajo |

### 6.2 LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Apply to model
model = AutoModelForCausalLM.from_pretrained("base_model")
model = get_peft_model(model, lora_config)

# Train
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)
trainer.train()

# Save only LoRA weights (tiny!)
model.save_pretrained("lora_weights")
```

### 6.3 QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)

# Apply LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Now can fine-tune 70B models on consumer GPU!
```

### 6.4 Instruction Tuning

```python
# Dataset format
instruction_dataset = [
    {
        "instruction": "Traduce al inglés",
        "input": "Hola mundo",
        "output": "Hello world"
    },
    {
        "instruction": "Resume el texto",
        "input": "Texto largo...",
        "output": "Resumen..."
    }
]

# Format for training
def format_instruction(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
```

### 6.5 RLHF (Reinforcement Learning from Human Feedback)

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. Train reward model
reward_model = train_reward_model(preference_data)

# 2. PPO training
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=16
)

model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
ppo_trainer = PPOTrainer(ppo_config, model, tokenizer)

# Training loop
for batch in dataloader:
    query_tensors = batch["input_ids"]

    # Generate
    response_tensors = ppo_trainer.generate(query_tensors)

    # Get reward
    rewards = [reward_model(q, r) for q, r in zip(queries, responses)]

    # PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

### 6.6 DPO (Direct Preference Optimization)

```python
from trl import DPOTrainer

# Preference data: chosen vs rejected
preference_data = [
    {
        "prompt": "Explain AI",
        "chosen": "Good explanation...",
        "rejected": "Poor explanation..."
    }
]

# DPO trainer (simpler than RLHF!)
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=preference_data,
    tokenizer=tokenizer
)

dpo_trainer.train()
```

### 6.7 Crear Datasets de Calidad

```python
# Data curation pipeline
def create_training_dataset():
    # 1. Collect raw data
    raw_data = collect_data(sources)

    # 2. Clean
    cleaned = remove_duplicates(raw_data)
    cleaned = filter_quality(cleaned, min_score=0.7)

    # 3. Augment
    augmented = back_translation(cleaned)
    augmented += paraphrase(cleaned)

    # 4. Format
    formatted = [format_example(ex) for ex in augmented]

    # 5. Split
    train, val, test = split_data(formatted, [0.8, 0.1, 0.1])

    return train, val, test
```

## Ejercicios

1. Fine-tune Llama 2 con LoRA para tarea específica
2. Implementar DPO para mejorar calidad de respuestas
3. Crear dataset de instrucciones para dominio especializado

[← Módulo Anterior](../modulo05_prompt_engineering/) | [Siguiente →](../modulo07_open_source_deployment/)
