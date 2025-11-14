# Módulo 4: IA Multimodal Avanzada

## Descripción

Los modelos multimodales de 2025 procesan y generan texto, imágenes, audio y video de forma unificada. Este módulo cubre las arquitecturas y aplicaciones más avanzadas.

## Contenido

### 4.1 Modelos Multimodales de Última Generación

**GPT-5 Multimodal**
- Procesamiento unificado de todas las modalidades
- Input/output simultáneo: texto + imagen + audio + video
- Tokenización compartida entre modalidades

**Gemini 2.5 Pro**
- "Native multimodal" desde el entrenamiento
- 1M+ tokens de contexto multimodal
- Razonamiento sobre imágenes complejas

**Claude 4 Vision**
- Análisis avanzado de imágenes
- Extracción de información de documentos visuales
- Diagramas, gráficos, screenshots

### 4.2 Vision-Language Models (VLMs)

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

# LLaVA-NeXT (2025)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

# Process image and text
image = Image.open("image.jpg")
prompt = "Describe esta imagen en detalle"

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

**Modelos VLM Destacados**:
- **LLaVA-NeXT**: Open source, 34B params
- **BLIP-2**: Efficient vision-language pretraining
- **CogVLM**: High-resolution image understanding
- **Qwen-VL**: Multilingual vision-language

### 4.3 Text-to-Image Generation

**DALL-E 3 (OpenAI)**
```python
from openai import OpenAI
client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="Un astronauta montando un caballo en Marte, estilo cyberpunk, 8k",
    size="1792x1024",
    quality="hd",
    n=1
)

image_url = response.data[0].url
```

**Midjourney V7 API**
```python
import midjourney

# Via API wrapper
mj = midjourney.Client(api_key="your_key")

image = mj.imagine(
    prompt="futuristic city, neon lights, rain --ar 16:9 --stylize 1000 --v 7",
    aspect_ratio="16:9"
)
```

**Stable Diffusion 3**
```python
from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe(
    prompt="A cat wearing a space suit floating in space",
    negative_prompt="blurry, low quality",
    num_inference_steps=40,
    guidance_scale=7.5
).images[0]

image.save("output.png")
```

### 4.4 Text-to-Video Generation

**Sora (OpenAI)**
```python
# Sora API (example)
from openai import OpenAI
client = OpenAI()

video = client.videos.generate(
    model="sora-1.0",
    prompt="A drone shot of waves crashing against cliffs at sunset",
    duration=10,  # seconds
    resolution="1080p"
)
```

**Runway Gen-3**
```python
import runway

client = runway.RunwayML(api_key="your_key")

task = client.videos.create(
    prompt="Time-lapse of a flower blooming",
    duration=5,
    style="cinematic"
)

video_url = task.wait_for_completion()
```

### 4.5 Text-to-Audio & Speech

**ElevenLabs**
```python
from elevenlabs import generate, play, voices

# List available voices
available_voices = voices()

# Generate speech
audio = generate(
    text="Bienvenido al curso avanzado de IA generativa",
    voice="Antoni",  # Or any voice ID
    model="eleven_multilingual_v2"
)

# Save
with open("output.mp3", "wb") as f:
    f.write(audio)
```

**OpenAI TTS**
```python
from openai import OpenAI
client = OpenAI()

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input="La IA multimodal está transformando cómo interactuamos con la tecnología"
)

response.stream_to_file("speech.mp3")
```

**Whisper (Speech-to-Text)**
```python
from openai import OpenAI
client = OpenAI()

audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)

print(transcript)
```

### 4.6 Multimodal RAG

```python
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import OpenCLIPEmbeddings

class MultimodalRAG:
    def __init__(self):
        # CLIP embeddings for images
        self.image_embeddings = OpenCLIPEmbeddings()

        # Text embeddings
        self.text_embeddings = OpenAIEmbeddings()

        # Separate vector stores
        self.image_store = Chroma(embedding_function=self.image_embeddings)
        self.text_store = Chroma(embedding_function=self.text_embeddings)

    def index_documents(self, documents):
        """Index both text and images"""
        for doc in documents:
            # Index text
            if doc.type == "text":
                self.text_store.add_documents([doc])

            # Index images with descriptions
            elif doc.type == "image":
                # Generate image description
                description = self.describe_image(doc.image_path)
                # Index image embedding + description
                self.image_store.add_documents([
                    Document(
                        page_content=description,
                        metadata={"image_path": doc.image_path}
                    )
                ])

    def retrieve(self, query: str, modality: str = "both"):
        """Retrieve from text, images, or both"""
        results = []

        if modality in ["text", "both"]:
            text_results = self.text_store.similarity_search(query, k=5)
            results.extend(text_results)

        if modality in ["image", "both"]:
            image_results = self.image_store.similarity_search(query, k=5)
            results.extend(image_results)

        return results

    def describe_image(self, image_path: str) -> str:
        """Generate image description using VLM"""
        # Use GPT-4V or similar
        with open(image_path, "rb") as f:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe esta imagen en detalle"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
        return response.choices[0].message.content
```

### 4.7 Casos de Uso Avanzados

**Análisis de Documentos Multimodal**
```python
def analyze_document(pdf_path: str):
    """Analiza PDF con texto e imágenes"""
    # Extract pages as images
    pages = convert_pdf_to_images(pdf_path)

    analysis = []
    for i, page in enumerate(pages):
        # Analyze each page
        result = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extrae información clave de esta página"},
                    {"type": "image_url", "image_url": {"url": encode_image(page)}}
                ]
            }]
        )
        analysis.append({
            "page": i+1,
            "content": result.choices[0].message.content
        })

    return analysis
```

**Video Understanding**
```python
def analyze_video(video_path: str):
    """Analiza video extrayendo frames clave"""
    # Extract key frames
    frames = extract_key_frames(video_path, interval=2)  # Every 2 seconds

    # Analyze frames
    timeline = []
    for timestamp, frame in frames:
        description = analyze_image(frame)
        timeline.append({
            "timestamp": timestamp,
            "description": description
        })

    # Generate video summary
    summary = generate_video_summary(timeline)
    return summary
```

**Audio + Transcript Analysis**
```python
def analyze_podcast(audio_path: str):
    """Transcribe y analiza podcast"""
    # 1. Transcribe
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb")
    )

    # 2. Analyze transcript
    analysis = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"""Analiza este podcast:

{transcript}

Proporciona:
1. Resumen ejecutivo
2. Puntos clave
3. Timestamps de temas importantes
4. Insights principales
"""
        }]
    )

    return analysis.choices[0].message.content
```

## Ejercicios

1. Implementar sistema de análisis de imágenes médicas
2. Crear generador de videos educativos con Sora
3. Construir RAG multimodal para documentos técnicos
4. Desarrollar asistente de voz multilingüe

## Recursos

- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [Hugging Face Multimodal](https://huggingface.co/models?pipeline_tag=image-to-text)
- [LLaVA Project](https://llava-vl.github.io/)

[← Módulo Anterior](../modulo03_rag_avanzado/) | [Siguiente Módulo →](../modulo05_prompt_engineering/)
