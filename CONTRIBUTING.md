# Contribuyendo al Curso Avanzado de IA Generativa

¡Gracias por tu interés en contribuir al curso! Este documento proporciona guías para contribuciones.

## Formas de Contribuir

### 1. Reportar Errores

Si encuentras un error en el contenido, código o ejemplos:

- Abre un Issue en GitHub
- Describe el error claramente
- Incluye pasos para reproducirlo
- Indica el módulo y archivo afectado

### 2. Sugerir Mejoras

Para sugerir mejoras al contenido:

- Abre un Issue con la etiqueta "enhancement"
- Explica la mejora propuesta
- Justifica por qué sería valiosa

### 3. Añadir Contenido

Para contribuir con nuevo contenido:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-seccion`)
3. Realiza tus cambios
4. Asegúrate de que el código funcione
5. Actualiza la documentación
6. Commit con mensajes descriptivos
7. Push a tu fork
8. Abre un Pull Request

### 4. Mejorar Ejemplos de Código

Si encuentras formas de mejorar los ejemplos:

- Optimizaciones de rendimiento
- Mejor manejo de errores
- Código más limpio
- Mejores prácticas

Son todas contribuciones bienvenidas.

## Guías de Estilo

### Código Python

- Sigue PEP 8
- Usa type hints
- Incluye docstrings
- Añade comentarios explicativos
- Maneja excepciones apropiadamente

```python
def example_function(param: str, count: int = 5) -> list:
    """
    Descripción breve de la función.

    Args:
        param: Descripción del parámetro
        count: Número de iteraciones (default: 5)

    Returns:
        Lista con resultados

    Raises:
        ValueError: Si param está vacío
    """
    if not param:
        raise ValueError("param no puede estar vacío")

    results = []
    for i in range(count):
        results.append(f"{param}_{i}")

    return results
```

### Documentación Markdown

- Usa headers apropiadamente (H1 para título, H2 para secciones principales)
- Incluye ejemplos de código con syntax highlighting
- Añade enlaces a recursos externos
- Usa listas para mejor legibilidad
- Incluye diagramas cuando sea apropiado

### Commits

Formato de mensajes de commit:

```
tipo(ámbito): descripción breve

Descripción detallada si es necesaria

Fixes #123
```

Tipos:
- `feat`: Nueva característica
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Formateo, punto y coma faltante, etc.
- `refactor`: Refactorización de código
- `test`: Añadir tests
- `chore`: Mantenimiento

Ejemplos:
```
feat(modulo03): añadir ejemplo de GraphRAG
fix(modulo01): corregir ejemplo de cuantización
docs(readme): actualizar instrucciones de instalación
```

## Proceso de Review

1. **Apertura de PR**: Describe claramente los cambios
2. **Automated checks**: Asegúrate de que pasen
3. **Review**: Mantente disponible para responder comentarios
4. **Iteración**: Realiza cambios solicitados
5. **Merge**: Una vez aprobado, será mergeado

## Código de Conducta

- Sé respetuoso y profesional
- Acepta críticas constructivas
- Enfócate en lo mejor para el curso
- Ayuda a otros contribuidores

## Preguntas

Si tienes preguntas sobre cómo contribuir:

- Abre un Issue con la etiqueta "question"
- Únete a nuestro Discord
- Envía email a contribuciones@curso-ia.com

## Reconocimiento

Todos los contribuidores serán reconocidos en:
- README.md
- CONTRIBUTORS.md
- Notas de release

¡Gracias por hacer este curso mejor para todos!
