"""
Módulo 2 - Ejemplo 1: Implementación de Agente ReAct
Demuestra cómo construir un agente que alterna entre razonamiento y acción.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Tool:
    """Definición de una herramienta que el agente puede usar"""
    name: str
    description: str
    function: callable


class ReActAgent:
    """Implementación de un agente ReAct (Reasoning + Acting)"""

    def __init__(self, tools: List[Tool], model: str = "gpt-4"):
        self.tools = {tool.name: tool for tool in tools}
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.max_iterations = 10

    def run(self, task: str) -> str:
        """Ejecutar el agente en una tarea"""
        print(f"\n{'='*80}")
        print(f"TAREA: {task}")
        print(f"{'='*80}\n")

        conversation_history = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Tarea: {task}"}
        ]

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteración {iteration + 1} ---")

            # Obtener respuesta del LLM
            response = self._get_llm_response(conversation_history)

            # Parsear respuesta
            thought, action, action_input = self._parse_response(response)

            # Mostrar razonamiento
            if thought:
                print(f"💭 Pensamiento: {thought}")

            # Si hay una acción, ejecutarla
            if action:
                print(f"🔧 Acción: {action}({action_input})")

                if action == "Final Answer":
                    print(f"\n{'='*80}")
                    print(f"✅ RESPUESTA FINAL: {action_input}")
                    print(f"{'='*80}\n")
                    return action_input

                # Ejecutar herramienta
                observation = self._execute_tool(action, action_input)
                print(f"👁️  Observación: {observation}")

                # Añadir a historial
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observación: {observation}"
                })
            else:
                # Si no hay acción clara, añadir y continuar
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

        return "No se pudo completar la tarea en el máximo de iteraciones."

    def _get_system_prompt(self) -> str:
        """Generar el prompt del sistema con descripción de herramientas"""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        return f"""Eres un agente ReAct que resuelve tareas alternando entre razonamiento y acción.

Herramientas disponibles:
{tools_desc}

Formato de respuesta:
Pensamiento: [tu razonamiento sobre qué hacer]
Acción: [nombre de la herramienta o "Final Answer"]
Entrada de Acción: [input para la herramienta o respuesta final]

Ejemplo:
Pensamiento: Necesito buscar información sobre Python
Acción: search
Entrada de Acción: Python programming language

Cuando tengas la respuesta final:
Pensamiento: Ya tengo toda la información necesaria
Acción: Final Answer
Entrada de Acción: [tu respuesta completa]

Importante:
- Usa las herramientas cuando sea necesario
- Razona antes de actuar
- Cuando tengas la respuesta, usa "Final Answer"
"""

    def _get_llm_response(self, conversation_history: List[Dict]) -> str:
        """Obtener respuesta del LLM"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation_history,
            temperature=0.2
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parsear la respuesta del LLM para extraer pensamiento, acción y entrada"""
        thought_match = re.search(r'Pensamiento:\s*(.+?)(?=\nAcción:|$)', response, re.DOTALL)
        action_match = re.search(r'Acción:\s*(.+?)(?=\n|$)', response)
        input_match = re.search(r'Entrada de Acción:\s*(.+?)(?=\n|$)', response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        action_input = input_match.group(1).strip() if input_match else None

        return thought, action, action_input

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Ejecutar una herramienta"""
        if tool_name not in self.tools:
            return f"Error: Herramienta '{tool_name}' no encontrada"

        try:
            result = self.tools[tool_name].function(tool_input)
            return str(result)
        except Exception as e:
            return f"Error al ejecutar herramienta: {str(e)}"


# ============================================================================
# DEFINICIÓN DE HERRAMIENTAS
# ============================================================================

def search_tool(query: str) -> str:
    """Simula una búsqueda web (en producción, usar API real)"""
    # Simulación de resultados de búsqueda
    knowledge_base = {
        "python": "Python es un lenguaje de programación de alto nivel, interpretado y de propósito general. Creado por Guido van Rossum en 1991.",
        "ai": "La Inteligencia Artificial es la simulación de procesos de inteligencia humana por sistemas informáticos.",
        "react": "ReAct es un paradigma que combina razonamiento (Reasoning) y acción (Acting) en agentes de IA.",
        "gpt": "GPT (Generative Pre-trained Transformer) es una familia de modelos de lenguaje desarrollados por OpenAI."
    }

    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value

    return f"Resultados de búsqueda para '{query}': No se encontró información específica."


def calculator_tool(expression: str) -> str:
    """Calculadora para operaciones matemáticas"""
    try:
        # Limpiar y evaluar expresión
        # NOTA: eval() es peligroso en producción, usar parser seguro
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Resultado: {result}"
    except Exception as e:
        return f"Error en cálculo: {str(e)}"


def date_tool(query: str) -> str:
    """Proporciona información sobre fechas"""
    from datetime import datetime

    now = datetime.now()
    info = {
        "fecha_actual": now.strftime("%Y-%m-%d"),
        "hora_actual": now.strftime("%H:%M:%S"),
        "día_semana": now.strftime("%A"),
        "mes": now.strftime("%B"),
        "año": now.year
    }

    return f"Información de fecha: {json.dumps(info, indent=2)}"


def python_repl_tool(code: str) -> str:
    """Ejecuta código Python (simplificado y seguro)"""
    # En producción, usar sandbox apropiado
    allowed_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "sum": sum,
            "max": max,
            "min": min,
        }
    }

    try:
        # Capturar output
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        exec(code, allowed_globals)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        return output if output else "Código ejecutado exitosamente (sin output)"
    except Exception as e:
        return f"Error al ejecutar código: {str(e)}"


# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

def main():
    """Ejemplos de uso del agente ReAct"""

    # Definir herramientas disponibles
    tools = [
        Tool(
            name="search",
            description="Busca información en internet. Input: consulta de búsqueda",
            function=search_tool
        ),
        Tool(
            name="calculator",
            description="Realiza cálculos matemáticos. Input: expresión matemática",
            function=calculator_tool
        ),
        Tool(
            name="date",
            description="Proporciona información sobre fechas. Input: tipo de información",
            function=date_tool
        ),
        Tool(
            name="python_repl",
            description="Ejecuta código Python. Input: código Python",
            function=python_repl_tool
        )
    ]

    # Crear agente
    agent = ReActAgent(tools=tools)

    # Ejemplo 1: Búsqueda de información
    print("\n" + "🤖 EJEMPLO 1: BÚSQUEDA Y RAZONAMIENTO".center(80, "="))
    agent.run("¿Qué es Python y para qué se usa?")

    # Ejemplo 2: Cálculos matemáticos
    print("\n" + "🤖 EJEMPLO 2: CÁLCULOS MATEMÁTICOS".center(80, "="))
    agent.run("Calcula el resultado de (150 + 75) * 2 / 5 y explica el proceso")

    # Ejemplo 3: Combinación de herramientas
    print("\n" + "🤖 EJEMPLO 3: MÚLTIPLES HERRAMIENTAS".center(80, "="))
    agent.run("Busca información sobre GPT, luego calcula cuántos años han pasado desde 2018 hasta hoy")

    # Ejemplo 4: Ejecución de código
    print("\n" + "🤖 EJEMPLO 4: EJECUCIÓN DE CÓDIGO".center(80, "="))
    agent.run("Crea una lista de números del 1 al 10 y calcula su suma usando Python")


if __name__ == "__main__":
    main()
