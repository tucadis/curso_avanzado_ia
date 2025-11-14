# Módulo 11: Casos de Uso Empresariales Avanzados

## Contenido

### 11.1 Asistentes Empresariales Inteligentes

```python
class EnterpriseAssistant:
    def __init__(self):
        self.tools = [
            self.search_documents,
            self.query_database,
            self.send_email,
            self.create_ticket,
            self.schedule_meeting
        ]
        self.memory = ConversationBufferMemory()

    async def handle_request(self, user_input: str, user_context: dict):
        # Understand intent
        intent = self.classify_intent(user_input)

        # Route to appropriate handler
        if intent == "information_retrieval":
            return await self.retrieve_information(user_input)
        elif intent == "task_execution":
            return await self.execute_task(user_input, user_context)
        elif intent == "analysis":
            return await self.perform_analysis(user_input)

    async def retrieve_information(self, query: str):
        # RAG pipeline
        docs = self.vector_store.similarity_search(query)
        response = self.llm.generate(
            f"Based on: {docs}\nAnswer: {query}"
        )
        return response
```

### 11.2 Code Generation & Copilots

```python
class CodeCopilot:
    def __init__(self):
        self.model = "claude-sonnet-4.5"  # Best for coding

    def generate_function(self, description: str, language: str = "python"):
        prompt = f"""
        Generate a {language} function that:
        {description}

        Requirements:
        - Include type hints
        - Add comprehensive docstring
        - Handle edge cases
        - Include error handling
        - Add unit tests

        Code:
        """

        code = self.llm.generate(prompt)
        return code

    def review_code(self, code: str):
        review_prompt = f"""
        Review this code for:
        1. Bugs and errors
        2. Performance issues
        3. Security vulnerabilities
        4. Best practices
        5. Code style

        Code:
        ```
        {code}
        ```

        Detailed review:
        """

        review = self.llm.generate(review_prompt)
        return review

    def refactor_code(self, code: str, goal: str):
        refactor_prompt = f"""
        Refactor this code to: {goal}

        Original code:
        ```
        {code}
        ```

        Refactored code:
        """

        refactored = self.llm.generate(refactor_prompt)
        return refactored
```

### 11.3 Análisis de Documentos

```python
class DocumentAnalyzer:
    def __init__(self):
        self.ocr = TesseractOCR()
        self.vlm = GPT4Vision()

    async def analyze_contract(self, pdf_path: str):
        # Extract text
        text = self.extract_text(pdf_path)

        # Extract clauses
        clauses = await self.extract_clauses(text)

        # Identify risks
        risks = await self.identify_risks(clauses)

        # Generate summary
        summary = await self.summarize_contract(text)

        return {
            "summary": summary,
            "key_clauses": clauses,
            "risks": risks,
            "recommendations": await self.generate_recommendations(risks)
        }

    async def extract_clauses(self, text: str):
        prompt = """
        Extract key clauses from this contract:
        - Payment terms
        - Termination conditions
        - Liability limitations
        - Confidentiality terms
        - Dispute resolution

        Contract:
        {text}

        Extracted clauses (JSON):
        """

        clauses = await self.llm.generate(prompt)
        return json.loads(clauses)
```

### 11.4 Customer Support Automation

```python
class CustomerSupportBot:
    def __init__(self):
        self.knowledge_base = load_knowledge_base()
        self.ticket_system = TicketSystem()

    async def handle_inquiry(self, customer_message: str, customer_id: str):
        # Retrieve customer history
        history = self.get_customer_history(customer_id)

        # Classify urgency
        urgency = self.classify_urgency(customer_message)

        # Attempt to resolve
        if urgency == "low":
            # Try automated resolution
            response = await self.generate_response(
                message=customer_message,
                context=history
            )

            # Check if satisfactory
            confidence = self.assess_confidence(response)

            if confidence > 0.8:
                return response
            else:
                # Escalate to human
                ticket = self.ticket_system.create_ticket(
                    customer_id=customer_id,
                    message=customer_message,
                    draft_response=response
                )
                return f"Ticket created: {ticket.id}"

        else:
            # High urgency - immediate escalation
            return self.escalate_to_human(customer_message, customer_id)
```

### 11.5 Content Generation at Scale

```python
class ContentGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.seo_optimizer = SEOOptimizer()

    async def generate_blog_post(
        self,
        topic: str,
        keywords: List[str],
        target_length: int = 1500
    ):
        # 1. Research
        research = await self.research_topic(topic)

        # 2. Create outline
        outline = await self.create_outline(topic, research)

        # 3. Write sections
        sections = []
        for section in outline:
            content = await self.write_section(
                section=section,
                context=research,
                keywords=keywords
            )
            sections.append(content)

        # 4. Combine
        full_content = "\n\n".join(sections)

        # 5. SEO optimize
        optimized = self.seo_optimizer.optimize(
            content=full_content,
            keywords=keywords
        )

        # 6. Generate meta
        meta = await self.generate_meta_data(optimized, keywords)

        return {
            "content": optimized,
            "meta_title": meta["title"],
            "meta_description": meta["description"],
            "keywords": keywords
        }

    async def batch_generate(self, topics: List[str]):
        # Parallel generation
        tasks = [self.generate_blog_post(topic) for topic in topics]
        results = await asyncio.gather(*tasks)
        return results
```

### 11.6 Data Analysis & BI

```python
class DataAnalystAgent:
    def __init__(self):
        self.code_executor = PythonREPL()

    async def analyze_data(self, data_path: str, question: str):
        # 1. Understand data
        data_summary = self.get_data_summary(data_path)

        # 2. Generate analysis code
        analysis_code = await self.generate_analysis_code(
            data_summary=data_summary,
            question=question
        )

        # 3. Execute code
        result = self.code_executor.run(analysis_code)

        # 4. Interpret results
        interpretation = await self.interpret_results(
            result=result,
            question=question
        )

        # 5. Generate visualization
        viz_code = await self.generate_visualization(result)
        chart = self.code_executor.run(viz_code)

        return {
            "analysis": interpretation,
            "chart": chart,
            "code": analysis_code
        }

    async def generate_analysis_code(self, data_summary: dict, question: str):
        prompt = f"""
        Generate Python code to analyze this data and answer the question.

        Data summary:
        {data_summary}

        Question: {question}

        Requirements:
        - Use pandas for analysis
        - Handle missing values
        - Include statistical tests if relevant
        - Return results as dictionary

        Code:
        ```python
        import pandas as pd
        import numpy as np

        # Your code here
        ```
        """

        code = await self.llm.generate(prompt)
        return extract_code(code)
```

### 11.7 Knowledge Management

```python
class KnowledgeManagementSystem:
    def __init__(self):
        self.vector_store = PineconeVectorStore()
        self.graph_db = Neo4jGraph()

    async def ingest_document(self, document: Document):
        # 1. Extract entities and relationships
        entities = await self.extract_entities(document.content)
        relationships = await self.extract_relationships(document.content)

        # 2. Store in graph
        for entity in entities:
            self.graph_db.add_entity(entity)

        for rel in relationships:
            self.graph_db.add_relationship(rel)

        # 3. Chunk and embed
        chunks = self.chunk_document(document)
        embeddings = self.embed_chunks(chunks)

        # 4. Store vectors
        self.vector_store.add_documents(chunks, embeddings)

    async def query(self, question: str):
        # 1. Hybrid search
        vector_results = self.vector_store.search(question)
        graph_results = self.graph_db.query(question)

        # 2. Combine contexts
        context = self.combine_contexts(vector_results, graph_results)

        # 3. Generate answer
        answer = await self.llm.generate(
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )

        return {
            "answer": answer,
            "sources": vector_results,
            "related_concepts": graph_results
        }
```

## Ejercicios

1. Implementar asistente empresarial con integración a Slack
2. Crear code copilot para framework específico
3. Desarrollar sistema de análisis de contratos
4. Construir bot de soporte con escalación inteligente

[← Módulo Anterior](../modulo10_observabilidad_mlops/) | [Siguiente →](../modulo12_sostenibilidad/)
