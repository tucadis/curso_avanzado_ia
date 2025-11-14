# Módulo 13: Aspectos Éticos, Legales y Regulatorios

## Contenido

### 13.1 Sesgos en Modelos de IA

**Detección de Sesgos**
```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

def detect_bias(predictions, protected_attribute):
    """Detecta sesgos en predicciones"""

    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=predictions,
        label_names=['prediction'],
        protected_attribute_names=[protected_attribute]
    )

    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=[{protected_attribute: 0}],
        privileged_groups=[{protected_attribute: 1}]
    )

    return {
        "disparate_impact": metric.disparate_impact(),
        "statistical_parity": metric.statistical_parity_difference(),
        "equal_opportunity": metric.equal_opportunity_difference()
    }
```

**Mitigación de Sesgos**
```python
def mitigate_bias_in_prompt(prompt: str) -> str:
    """Añade instrucciones para reducir sesgos"""

    debiased_prompt = f"""
    {prompt}

    IMPORTANTE: Proporciona una respuesta objetiva y libre de sesgos.
    - No hagas suposiciones basadas en género, raza, edad, etc.
    - Trata a todos los grupos demográficos equitativamente
    - Si no tienes información suficiente, indícalo
    - Evita estereotipos y generalizaciones

    Respuesta:
    """

    return debiased_prompt
```

### 13.2 Fairness & Equidad

**Fairness Metrics**
```python
class FairnessEvaluator:
    def evaluate_fairness(
        self,
        predictions: pd.DataFrame,
        protected_attrs: List[str]
    ) -> dict:
        """Evalúa fairness en múltiples dimensiones"""

        results = {}

        for attr in protected_attrs:
            # Demographic Parity
            demo_parity = self.demographic_parity(predictions, attr)

            # Equal Opportunity
            eq_opp = self.equal_opportunity(predictions, attr)

            # Calibration
            calibration = self.calibration(predictions, attr)

            results[attr] = {
                "demographic_parity": demo_parity,
                "equal_opportunity": eq_opp,
                "calibration": calibration,
                "passes_threshold": all([
                    demo_parity > 0.8,
                    eq_opp > 0.8,
                    calibration > 0.9
                ])
            }

        return results
```

### 13.3 AI Act Europeo (Compliance)

**Clasificación de Riesgo**
```python
class AIActCompliance:
    RISK_LEVELS = {
        "unacceptable": [
            "social_scoring",
            "subliminal_manipulation",
            "exploitation_vulnerabilities"
        ],
        "high": [
            "biometric_identification",
            "critical_infrastructure",
            "education_scoring",
            "employment_decisions",
            "law_enforcement",
            "migration_decisions"
        ],
        "limited": [
            "chatbots",
            "emotion_recognition",
            "deepfakes"
        ],
        "minimal": [
            "spam_filters",
            "inventory_management"
        ]
    }

    def classify_application(self, use_case: str) -> dict:
        """Clasifica aplicación según AI Act"""

        for level, cases in self.RISK_LEVELS.items():
            if use_case in cases:
                return {
                    "risk_level": level,
                    "requirements": self.get_requirements(level),
                    "prohibited": level == "unacceptable"
                }

        return {"risk_level": "minimal", "requirements": []}

    def get_requirements(self, risk_level: str) -> List[str]:
        requirements = {
            "high": [
                "Risk management system",
                "Data governance",
                "Technical documentation",
                "Logging capabilities",
                "Human oversight",
                "Accuracy requirements",
                "Robustness testing",
                "Cybersecurity measures"
            ],
            "limited": [
                "Transparency obligations",
                "User notification",
                "Disclosure of AI use"
            ]
        }

        return requirements.get(risk_level, [])
```

**Documentation Requirements**
```python
class AISystemDocumentation:
    def generate_compliance_docs(self, system_info: dict):
        """Genera documentación para AI Act compliance"""

        docs = {
            "technical_documentation": self.technical_docs(system_info),
            "risk_assessment": self.risk_assessment(system_info),
            "data_governance": self.data_governance_plan(system_info),
            "human_oversight": self.oversight_measures(system_info),
            "transparency_info": self.transparency_docs(system_info)
        }

        return docs

    def technical_docs(self, info: dict):
        return {
            "system_description": info["description"],
            "intended_purpose": info["purpose"],
            "development_process": info["development"],
            "data_requirements": info["data"],
            "performance_metrics": info["metrics"],
            "limitations": info["limitations"],
            "monitoring_procedures": info["monitoring"]
        }
```

### 13.4 Derechos de Autor y Propiedad Intelectual

**Copyright Compliance**
```python
class CopyrightChecker:
    def check_training_data(self, dataset: List[str]) -> dict:
        """Verifica compliance de datos de entrenamiento"""

        issues = []

        for doc in dataset:
            # Check for copyrighted material
            if self.is_copyrighted(doc):
                issues.append({
                    "document": doc,
                    "issue": "Potential copyright violation",
                    "recommendation": "Remove or obtain license"
                })

        return {
            "total_docs": len(dataset),
            "issues_found": len(issues),
            "compliant": len(issues) == 0,
            "details": issues
        }

    def check_generated_content(self, content: str) -> dict:
        """Verifica si contenido generado infringe copyright"""

        # Check for exact matches with known copyrighted works
        similarity = self.check_similarity_to_known_works(content)

        return {
            "original": similarity < 0.7,  # Less than 70% similarity
            "similarity_score": similarity,
            "potential_sources": self.find_similar_sources(content)
        }
```

### 13.5 Transparencia y Explicabilidad (XAI)

**Explainable AI**
```python
class ExplainableAI:
    def explain_decision(
        self,
        input_data: str,
        output: str,
        model: str
    ) -> dict:
        """Genera explicación de decisión del modelo"""

        explanation_prompt = f"""
        Explica por qué el modelo {model} generó esta respuesta:

        Input: {input_data}
        Output: {output}

        Proporciona:
        1. Factores clave que influyeron en la respuesta
        2. Información del input que fue más relevante
        3. Alternativas consideradas
        4. Nivel de confianza

        Explicación:
        """

        explanation = self.meta_llm.generate(explanation_prompt)

        return {
            "explanation": explanation,
            "confidence_score": self.calculate_confidence(input_data, output),
            "key_factors": self.extract_factors(explanation),
            "transparency_level": "high"
        }

    def generate_transparency_card(self, model_info: dict):
        """Genera Model Card para transparencia"""

        card = {
            "model_details": {
                "name": model_info["name"],
                "version": model_info["version"],
                "type": model_info["type"],
                "owner": model_info["owner"]
            },
            "intended_use": {
                "primary_uses": model_info["uses"],
                "out_of_scope": model_info["not_for"]
            },
            "factors": {
                "groups": model_info["demographic_groups"],
                "instrumentation": model_info["data_collection"],
                "environment": model_info["deployment_context"]
            },
            "metrics": {
                "performance": model_info["performance_metrics"],
                "decision_thresholds": model_info["thresholds"]
            },
            "training_data": {
                "description": model_info["data_description"],
                "preprocessing": model_info["preprocessing"]
            },
            "ethical_considerations": {
                "bias_analysis": model_info["bias_assessment"],
                "fairness_metrics": model_info["fairness"],
                "privacy_measures": model_info["privacy"]
            },
            "caveats_recommendations": model_info["limitations"]
        }

        return card
```

### 13.6 Privacidad y GDPR

**GDPR Compliance**
```python
class GDPRCompliance:
    def ensure_right_to_erasure(self, user_id: str):
        """Implementa derecho al olvido (Art. 17 GDPR)"""

        # Delete user data
        self.database.delete_user_data(user_id)

        # Remove from training data
        self.remove_from_training_data(user_id)

        # Purge from caches
        self.clear_caches(user_id)

        # Log deletion
        self.audit_log.record_deletion(user_id, timestamp=datetime.now())

    def implement_data_minimization(self, data: dict) -> dict:
        """Minimiza datos recolectados (Art. 5 GDPR)"""

        # Only collect necessary data
        necessary_fields = ["user_id", "query", "timestamp"]

        minimized = {k: v for k, v in data.items() if k in necessary_fields}

        # Anonymize if possible
        minimized["user_id"] = self.anonymize_id(minimized["user_id"])

        return minimized

    def handle_data_access_request(self, user_id: str):
        """Implementa derecho de acceso (Art. 15 GDPR)"""

        user_data = {
            "personal_data": self.get_user_data(user_id),
            "processing_purposes": self.get_purposes(),
            "data_categories": self.get_categories(),
            "recipients": self.get_recipients(),
            "retention_period": self.get_retention_period(),
            "rights": self.explain_rights()
        }

        return user_data
```

### 13.7 Consideraciones Éticas

**Ethical AI Framework**
```python
class EthicalAIFramework:
    PRINCIPLES = [
        "beneficence",  # Do good
        "non_maleficence",  # Do no harm
        "autonomy",  # Respect user autonomy
        "justice",  # Fair treatment
        "explicability"  # Transparency
    ]

    def ethical_review(self, use_case: dict) -> dict:
        """Review ético de caso de uso"""

        review = {}

        # Beneficence
        review["beneficence"] = self.assess_benefit(use_case)

        # Non-maleficence
        review["risks"] = self.identify_potential_harms(use_case)

        # Autonomy
        review["user_control"] = self.assess_user_autonomy(use_case)

        # Justice
        review["fairness"] = self.assess_fairness(use_case)

        # Explicability
        review["transparency"] = self.assess_transparency(use_case)

        # Overall assessment
        review["approved"] = all([
            review["beneficence"]["score"] > 0.7,
            len(review["risks"]["high_risk"]) == 0,
            review["user_control"]["adequate"],
            review["fairness"]["passes_requirements"],
            review["transparency"]["sufficient"]
        ])

        return review

    def identify_potential_harms(self, use_case: dict) -> dict:
        """Identifica potenciales daños"""

        harms = {
            "privacy_risks": [],
            "bias_risks": [],
            "security_risks": [],
            "manipulation_risks": [],
            "dependency_risks": []
        }

        # Analyze use case for each risk category
        # ...

        return harms
```

## Ejercicios

1. Implementar detector de sesgos para modelo
2. Crear documentación completa para AI Act compliance
3. Desarrollar sistema de explicabilidad para decisiones
4. Implementar GDPR compliance tools

[← Módulo Anterior](../modulo12_sostenibilidad/) | [Siguiente →](../modulo14_tecnicas_avanzadas/)
