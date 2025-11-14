"""
Módulo 1 - Ejemplo 1: Comparación de Modelos LLM
Este script demuestra cómo usar diferentes modelos (GPT-5, Claude Sonnet 4.5, Gemini 2.5)
y comparar sus respuestas en la misma tarea.
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# Cargar variables de entorno
load_dotenv()

class ModelComparator:
    """Clase para comparar respuestas de diferentes modelos LLM"""

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def query_gpt5(self, prompt: str, **kwargs) -> dict:
        """Consultar GPT-5"""
        start_time = time.time()

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # o el modelo específico disponible
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en IA."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            )

            elapsed_time = time.time() - start_time

            return {
                "model": "GPT-5",
                "response": response.choices[0].message.content,
                "time": elapsed_time,
                "tokens_input": response.usage.prompt_tokens,
                "tokens_output": response.usage.completion_tokens,
                "cost_estimate": self._calculate_cost_gpt5(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            }
        except Exception as e:
            return {
                "model": "GPT-5",
                "error": str(e),
                "time": time.time() - start_time
            }

    def query_claude_sonnet(self, prompt: str, **kwargs) -> dict:
        """Consultar Claude Sonnet 4.5"""
        start_time = time.time()

        try:
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4.5",
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            elapsed_time = time.time() - start_time

            return {
                "model": "Claude Sonnet 4.5",
                "response": message.content[0].text,
                "time": elapsed_time,
                "tokens_input": message.usage.input_tokens,
                "tokens_output": message.usage.output_tokens,
                "cost_estimate": self._calculate_cost_claude(
                    message.usage.input_tokens,
                    message.usage.output_tokens
                )
            }
        except Exception as e:
            return {
                "model": "Claude Sonnet 4.5",
                "error": str(e),
                "time": time.time() - start_time
            }

    def query_gemini(self, prompt: str, **kwargs) -> dict:
        """Consultar Gemini 2.5 Pro"""
        start_time = time.time()

        try:
            model = genai.GenerativeModel('gemini-2.5-pro')

            generation_config = genai.GenerationConfig(
                temperature=kwargs.get("temperature", 0.7),
                max_output_tokens=kwargs.get("max_tokens", 1000)
            )

            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )

            elapsed_time = time.time() - start_time

            return {
                "model": "Gemini 2.5 Pro",
                "response": response.text,
                "time": elapsed_time,
                "tokens_input": response.usage_metadata.prompt_token_count,
                "tokens_output": response.usage_metadata.candidates_token_count,
                "cost_estimate": self._calculate_cost_gemini(
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count
                )
            }
        except Exception as e:
            return {
                "model": "Gemini 2.5 Pro",
                "error": str(e),
                "time": time.time() - start_time
            }

    def _calculate_cost_gpt5(self, input_tokens: int, output_tokens: int) -> float:
        """Calcular costo estimado para GPT-5"""
        # Precios aproximados (ajustar según pricing real)
        INPUT_PRICE = 30.0 / 1_000_000  # $30 per 1M tokens
        OUTPUT_PRICE = 60.0 / 1_000_000  # $60 per 1M tokens
        return (input_tokens * INPUT_PRICE) + (output_tokens * OUTPUT_PRICE)

    def _calculate_cost_claude(self, input_tokens: int, output_tokens: int) -> float:
        """Calcular costo estimado para Claude Sonnet 4.5"""
        INPUT_PRICE = 3.0 / 1_000_000  # $3 per 1M tokens
        OUTPUT_PRICE = 15.0 / 1_000_000  # $15 per 1M tokens
        return (input_tokens * INPUT_PRICE) + (output_tokens * OUTPUT_PRICE)

    def _calculate_cost_gemini(self, input_tokens: int, output_tokens: int) -> float:
        """Calcular costo estimado para Gemini 2.5 Pro"""
        INPUT_PRICE = 2.5 / 1_000_000  # $2.5 per 1M tokens
        OUTPUT_PRICE = 7.0 / 1_000_000  # $7 per 1M tokens
        return (input_tokens * INPUT_PRICE) + (output_tokens * OUTPUT_PRICE)

    def compare_all(self, prompt: str, **kwargs) -> dict:
        """Comparar todos los modelos con el mismo prompt"""
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*80}\n")

        results = {
            "gpt5": self.query_gpt5(prompt, **kwargs),
            "claude": self.query_claude_sonnet(prompt, **kwargs),
            "gemini": self.query_gemini(prompt, **kwargs)
        }

        # Mostrar resultados
        for model_key, result in results.items():
            print(f"\n{'-'*80}")
            print(f"MODELO: {result.get('model', model_key.upper())}")
            print(f"{'-'*80}")

            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"RESPUESTA:\n{result['response']}\n")
                print(f"Tiempo: {result['time']:.2f}s")
                print(f"Tokens (input/output): {result['tokens_input']}/{result['tokens_output']}")
                print(f"Costo estimado: ${result['cost_estimate']:.6f}")

        # Resumen comparativo
        print(f"\n{'='*80}")
        print("RESUMEN COMPARATIVO")
        print(f"{'='*80}")

        valid_results = {k: v for k, v in results.items() if "error" not in v}

        if valid_results:
            fastest = min(valid_results.items(), key=lambda x: x[1]['time'])
            cheapest = min(valid_results.items(), key=lambda x: x[1]['cost_estimate'])

            print(f"Más rápido: {valid_results[fastest[0]]['model']} ({fastest[1]['time']:.2f}s)")
            print(f"Más económico: {valid_results[cheapest[0]]['model']} (${cheapest[1]['cost_estimate']:.6f})")

        return results


def main():
    """Función principal con ejemplos de uso"""
    comparator = ModelComparator()

    # Ejemplo 1: Tarea de razonamiento
    print("\n🧠 EJEMPLO 1: RAZONAMIENTO LÓGICO")
    prompt1 = """
    Resuelve el siguiente problema paso a paso:

    Si un tren sale de la ciudad A hacia la ciudad B a 80 km/h, y otro tren
    sale de B hacia A a 100 km/h, y la distancia entre ciudades es de 450 km,
    ¿en cuánto tiempo se encontrarán y a qué distancia de A?
    """
    comparator.compare_all(prompt1, temperature=0.2)

    # Ejemplo 2: Generación de código
    print("\n\n💻 EJEMPLO 2: GENERACIÓN DE CÓDIGO")
    prompt2 = """
    Escribe una función en Python que implemente el algoritmo de búsqueda binaria
    de forma recursiva. Incluye docstrings y type hints.
    """
    comparator.compare_all(prompt2, temperature=0.3)

    # Ejemplo 3: Análisis creativo
    print("\n\n🎨 EJEMPLO 3: ANÁLISIS CREATIVO")
    prompt3 = """
    Analiza las diferencias entre arquitecturas de transformers y state space models
    desde la perspectiva de eficiencia computacional. Proporciona un ejemplo de
    cuándo usar cada una.
    """
    comparator.compare_all(prompt3, temperature=0.7)


if __name__ == "__main__":
    main()
