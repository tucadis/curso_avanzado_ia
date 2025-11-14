# Módulo 14: Técnicas Avanzadas y Fronteras de Investigación

## Contenido

### 14.1 Mixture of Experts (MoE)

```python
# Conceptual MoE architecture
class MixtureOfExperts:
    def __init__(self, num_experts: int = 8, top_k: int = 2):
        self.experts = [ExpertModel(i) for i in range(num_experts)]
        self.router = RouterNetwork()
        self.top_k = top_k

    def forward(self, x):
        # Router decides which experts to use
        expert_weights = self.router(x)  # Shape: [batch, num_experts]

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k)

        # Normalize weights
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # Combine expert outputs
        output = 0
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i]

            expert_output = self.experts[expert_idx](x)
            output += weight * expert_output

        return output

# Benefits:
# - Scales to trillions of parameters
# - Only activates subset (sparse)
# - Specialized experts for different tasks
# - Used in GPT-4, Mixtral, etc.
```

### 14.2 Constitutional AI

```python
class ConstitutionalAI:
    """Implementa Constitutional AI (Anthropic)"""

    def __init__(self):
        self.constitution = self.load_constitution()
        self.base_model = load_model("base")
        self.critic_model = load_model("critic")

    def load_constitution(self):
        return [
            "Be helpful and harmless",
            "Respect human autonomy",
            "Be truthful and accurate",
            "Protect privacy",
            "Avoid bias and discrimination",
            "Refuse harmful requests"
        ]

    def generate_with_constitution(self, prompt: str):
        # 1. Initial generation
        initial_response = self.base_model.generate(prompt)

        # 2. Critique against constitution
        critique = self.critique_response(initial_response)

        # 3. Revise if needed
        if critique["violations"]:
            revised_prompt = f"""
            Original: {initial_response}

            Critique: {critique["feedback"]}

            Revise to align with principles:
            {self.constitution}

            Revised:
            """

            final_response = self.base_model.generate(revised_prompt)
            return final_response

        return initial_response

    def critique_response(self, response: str):
        critique_prompt = f"""
        Evaluate this response against these principles:
        {json.dumps(self.constitution, indent=2)}

        Response: {response}

        Violations (if any):
        """

        critique = self.critic_model.generate(critique_prompt)

        return {
            "feedback": critique,
            "violations": self.parse_violations(critique)
        }
```

### 14.3 Reasoning Models & Test-Time Compute

**Chain-of-Thought at Scale**
```python
class O1StyleReasoning:
    """Reasoning model that uses test-time compute"""

    def __init__(self):
        self.model = load_model("reasoning-model")

    def solve_with_reasoning(self, problem: str, compute_budget: int = 10):
        """Uses more compute at inference for better reasoning"""

        best_solution = None
        best_confidence = 0

        for attempt in range(compute_budget):
            # Generate reasoning chain
            reasoning = self.generate_reasoning_chain(
                problem,
                temperature=0.7,  # Sample diverse approaches
                max_tokens=2000
            )

            # Generate solution based on reasoning
            solution = self.generate_solution(reasoning)

            # Verify solution
            confidence = self.verify_solution(problem, solution, reasoning)

            if confidence > best_confidence:
                best_solution = {
                    "reasoning": reasoning,
                    "solution": solution,
                    "confidence": confidence
                }
                best_confidence = confidence

            # Early stopping if high confidence
            if confidence > 0.95:
                break

        return best_solution

    def generate_reasoning_chain(self, problem: str, **kwargs):
        prompt = f"""
        Problem: {problem}

        Let's solve this step by step, showing all reasoning:

        Step 1:
        """

        return self.model.generate(prompt, **kwargs)
```

### 14.4 Continual Learning

```python
class ContinualLearningSystem:
    """Model that learns continuously without forgetting"""

    def __init__(self):
        self.model = load_base_model()
        self.memory = ExperienceReplay()
        self.ewc = ElasticWeightConsolidation()  # Prevent forgetting

    def learn_new_task(self, new_data, task_id: int):
        # 1. Store important examples in memory
        self.memory.store_examples(new_data, task_id)

        # 2. Calculate Fisher information (EWC)
        fisher_info = self.ewc.calculate_fisher(self.model, new_data)

        # 3. Fine-tune with regularization
        for batch in new_data:
            # Standard loss
            loss = self.model.compute_loss(batch)

            # EWC regularization (prevents forgetting)
            ewc_loss = self.ewc.penalty(self.model, fisher_info)

            total_loss = loss + ewc_loss

            # Update
            total_loss.backward()
            self.optimizer.step()

        # 4. Replay old tasks occasionally
        if len(self.memory) > 0:
            old_batch = self.memory.sample()
            self.model.train_on_batch(old_batch)

    def prevent_catastrophic_forgetting(self):
        """Strategies to maintain previous knowledge"""

        strategies = {
            "ewc": "Elastic Weight Consolidation",
            "replay": "Experience Replay",
            "progressive": "Progressive Neural Networks",
            "adapter": "Adapter Modules"
        }

        return strategies
```

### 14.5 Neurosymbolic AI

```python
class NeurosymbolicSystem:
    """Combines neural networks with symbolic reasoning"""

    def __init__(self):
        self.neural = NeuralComponent()  # LLM
        self.symbolic = SymbolicReasoner()  # Logic engine

    def solve_problem(self, problem: str):
        # 1. Neural: Extract structure from natural language
        structured = self.neural.extract_structure(problem)

        # 2. Symbolic: Apply logical reasoning
        solution = self.symbolic.reason(structured)

        # 3. Neural: Convert back to natural language
        explanation = self.neural.explain(solution)

        return {
            "solution": solution,
            "explanation": explanation,
            "confidence": self.verify_logical_consistency(solution)
        }

    def verify_logical_consistency(self, solution):
        """Use symbolic reasoning to verify"""
        return self.symbolic.verify(solution)

# Example: Math problem solving
class MathSolver(NeurosymbolicSystem):
    def solve_math_problem(self, problem: str):
        # Neural: Parse problem
        parsed = self.neural.parse(
            f"Convert to equations: {problem}"
        )

        # Symbolic: Solve equations
        solution = sympy.solve(parsed.equations)

        # Neural: Explain
        explanation = self.neural.generate(
            f"Explain solution: {solution}"
        )

        return {
            "answer": solution,
            "explanation": explanation,
            "verified": True  # Symbolic solver guarantees correctness
        }
```

### 14.6 World Models

```python
class WorldModel:
    """Model that builds internal representation of world"""

    def __init__(self):
        self.model = load_model()
        self.world_state = {}

    def simulate_action(self, state: dict, action: str):
        """Simulate outcome of action without executing"""

        prompt = f"""
        Current state:
        {json.dumps(state, indent=2)}

        Action: {action}

        Predict the outcome:
        - New state
        - Side effects
        - Probability of success

        Prediction:
        """

        prediction = self.model.generate(prompt)

        return self.parse_prediction(prediction)

    def plan_with_simulation(self, goal: str, current_state: dict):
        """Plan by simulating multiple paths"""

        def search(state, depth=0, max_depth=5):
            if self.goal_reached(state, goal):
                return []

            if depth >= max_depth:
                return None

            # Generate possible actions
            actions = self.generate_possible_actions(state)

            # Simulate each action
            best_plan = None
            best_score = -float('inf')

            for action in actions:
                # Simulate outcome
                new_state = self.simulate_action(state, action)

                # Recursively plan from new state
                rest_of_plan = search(new_state, depth + 1, max_depth)

                if rest_of_plan is not None:
                    score = self.evaluate_plan([action] + rest_of_plan)

                    if score > best_score:
                        best_plan = [action] + rest_of_plan
                        best_score = score

            return best_plan

        return search(current_state)
```

### 14.7 Memory Augmentation

**Long-term Memory Systems**
```python
class MemoryAugmentedLLM:
    def __init__(self):
        self.llm = load_model()
        self.short_term = ConversationBuffer()
        self.long_term = VectorMemory()
        self.episodic = EpisodicMemory()

    def remember(self, interaction: dict):
        # Store in appropriate memory
        self.short_term.add(interaction)

        # Important interactions → long-term
        if self.is_important(interaction):
            self.long_term.store(interaction)

        # Episodic memory for experiences
        self.episodic.store_episode(interaction)

    def recall(self, query: str):
        # Retrieve from all memory types
        short_term_context = self.short_term.get_recent(k=5)

        long_term_context = self.long_term.similarity_search(query, k=3)

        episodic_context = self.episodic.recall_similar_episodes(query)

        # Combine contexts
        full_context = {
            "recent": short_term_context,
            "knowledge": long_term_context,
            "experiences": episodic_context
        }

        return full_context

    def generate_with_memory(self, prompt: str):
        # Recall relevant memories
        context = self.recall(prompt)

        # Generate with full context
        augmented_prompt = self.build_prompt_with_context(prompt, context)

        response = self.llm.generate(augmented_prompt)

        # Remember this interaction
        self.remember({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now()
        })

        return response
```

### 14.8 Multimodal Reasoning

```python
class MultimodalReasoningSystem:
    """Advanced reasoning across modalities"""

    def __init__(self):
        self.vlm = VisionLanguageModel()
        self.reasoner = ReasoningModel()

    def solve_visual_reasoning(self, image, question: str):
        # 1. Perceive: Extract visual features
        visual_features = self.vlm.encode_image(image)

        # 2. Describe: Generate descriptions
        descriptions = self.vlm.describe(image)

        # 3. Reason: Apply logical reasoning
        reasoning_chain = self.reasoner.reason(
            visual_info=descriptions,
            question=question
        )

        # 4. Answer: Generate final answer
        answer = self.reasoner.conclude(reasoning_chain)

        return {
            "answer": answer,
            "reasoning": reasoning_chain,
            "visual_evidence": descriptions
        }
```

## Papers de Referencia

- "Mixture of Experts" - Shazeer et al.
- "Constitutional AI" - Bai et al., Anthropic
- "Let's Verify Step by Step" - OpenAI (process supervision)
- "Continual Learning Survey" - Parisi et al.
- "Neural-Symbolic Integration" - Garcez et al.

## Ejercicios

1. Implementar simple MoE router
2. Crear Constitutional AI system
3. Desarrollar world model simulator
4. Construir memory-augmented agent

[← Módulo Anterior](../modulo13_etica_legal/) | [Siguiente →](../modulo15_proyecto_final/)
