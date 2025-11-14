"""
Módulo 1 - Ejemplo 2: Cuantización de Modelos
Demuestra cómo cuantizar modelos usando diferentes técnicas para reducir tamaño y acelerar inferencia.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig
)
import time
import psutil
import os


class ModelQuantizer:
    """Clase para demostrar diferentes técnicas de cuantización"""

    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_full_precision(self):
        """Cargar modelo en precisión completa (FP32)"""
        print(f"\n{'='*60}")
        print("Cargando modelo en FP32 (Full Precision)")
        print(f"{'='*60}")

        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        load_time = time.time() - start_time

        model_size = self._get_model_size(model)
        memory_usage = self._get_memory_usage()

        print(f"✓ Tiempo de carga: {load_time:.2f}s")
        print(f"✓ Tamaño del modelo: {model_size:.2f} MB")
        print(f"✓ Memoria RAM usada: {memory_usage:.2f} MB")

        return model, {"load_time": load_time, "size_mb": model_size, "memory_mb": memory_usage}

    def load_fp16(self):
        """Cargar modelo en FP16 (Half Precision)"""
        print(f"\n{'='*60}")
        print("Cargando modelo en FP16 (Half Precision)")
        print(f"{'='*60}")

        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        load_time = time.time() - start_time

        model_size = self._get_model_size(model)
        memory_usage = self._get_memory_usage()

        print(f"✓ Tiempo de carga: {load_time:.2f}s")
        print(f"✓ Tamaño del modelo: {model_size:.2f} MB")
        print(f"✓ Memoria RAM usada: {memory_usage:.2f} MB")
        print(f"✓ Reducción vs FP32: ~50%")

        return model, {"load_time": load_time, "size_mb": model_size, "memory_mb": memory_usage}

    def load_8bit(self):
        """Cargar modelo cuantizado a 8 bits usando bitsandbytes"""
        print(f"\n{'='*60}")
        print("Cargando modelo en INT8 (8-bit quantization)")
        print(f"{'='*60}")

        start_time = time.time()

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        load_time = time.time() - start_time

        model_size = self._get_model_size(model)
        memory_usage = self._get_memory_usage()

        print(f"✓ Tiempo de carga: {load_time:.2f}s")
        print(f"✓ Tamaño del modelo: {model_size:.2f} MB")
        print(f"✓ Memoria RAM usada: {memory_usage:.2f} MB")
        print(f"✓ Reducción vs FP32: ~75%")

        return model, {"load_time": load_time, "size_mb": model_size, "memory_mb": memory_usage}

    def load_4bit(self):
        """Cargar modelo cuantizado a 4 bits usando bitsandbytes"""
        print(f"\n{'='*60}")
        print("Cargando modelo en INT4 (4-bit quantization)")
        print(f"{'='*60}")

        start_time = time.time()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        load_time = time.time() - start_time

        model_size = self._get_model_size(model)
        memory_usage = self._get_memory_usage()

        print(f"✓ Tiempo de carga: {load_time:.2f}s")
        print(f"✓ Tamaño del modelo: {model_size:.2f} MB")
        print(f"✓ Memoria RAM usada: {memory_usage:.2f} MB")
        print(f"✓ Reducción vs FP32: ~87.5%")

        return model, {"load_time": load_time, "size_mb": model_size, "memory_mb": memory_usage}

    def benchmark_inference(self, model, prompt: str, num_tokens: int = 50) -> dict:
        """Benchmark de velocidad de inferencia"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Warm-up
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)

        # Benchmark real
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=True,
                temperature=0.7
            )
        inference_time = time.time() - start_time

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_per_second = num_tokens / inference_time

        return {
            "inference_time": inference_time,
            "tokens_per_second": tokens_per_second,
            "generated_text": generated_text
        }

    def _get_model_size(self, model) -> float:
        """Obtener tamaño del modelo en MB"""
        param_size = sum([param.nelement() * param.element_size() for param in model.parameters()])
        buffer_size = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria del proceso actual"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

    def compare_all_quantizations(self, test_prompt: str):
        """Comparar todas las técnicas de cuantización"""
        results = {}

        try:
            print("\n🔬 COMPARACIÓN DE TÉCNICAS DE CUANTIZACIÓN")
            print("="*80)

            # FP16 (FP32 puede ser muy pesado para demo)
            model_fp16, stats_fp16 = self.load_fp16()
            bench_fp16 = self.benchmark_inference(model_fp16, test_prompt)
            results['fp16'] = {**stats_fp16, **bench_fp16}
            del model_fp16
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # INT8
            model_8bit, stats_8bit = self.load_8bit()
            bench_8bit = self.benchmark_inference(model_8bit, test_prompt)
            results['int8'] = {**stats_8bit, **bench_8bit}
            del model_8bit
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # INT4
            model_4bit, stats_4bit = self.load_4bit()
            bench_4bit = self.benchmark_inference(model_4bit, test_prompt)
            results['int4'] = {**stats_4bit, **bench_4bit}
            del model_4bit
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Mostrar comparación
            self._print_comparison_table(results)

        except Exception as e:
            print(f"\n❌ Error durante la comparación: {e}")

        return results

    def _print_comparison_table(self, results: dict):
        """Imprimir tabla comparativa de resultados"""
        print(f"\n{'='*80}")
        print("TABLA COMPARATIVA DE CUANTIZACIÓN")
        print(f"{'='*80}")
        print(f"{'Métrica':<25} {'FP16':<15} {'INT8':<15} {'INT4':<15}")
        print(f"{'-'*80}")

        # Tamaño del modelo
        fp16_size = results['fp16']['size_mb']
        int8_size = results['int8']['size_mb']
        int4_size = results['int4']['size_mb']
        print(f"{'Tamaño (MB)':<25} {fp16_size:<15.2f} {int8_size:<15.2f} {int4_size:<15.2f}")

        # Tiempo de carga
        print(f"{'Tiempo carga (s)':<25} {results['fp16']['load_time']:<15.2f} "
              f"{results['int8']['load_time']:<15.2f} {results['int4']['load_time']:<15.2f}")

        # Tokens por segundo
        print(f"{'Tokens/segundo':<25} {results['fp16']['tokens_per_second']:<15.2f} "
              f"{results['int8']['tokens_per_second']:<15.2f} {results['int4']['tokens_per_second']:<15.2f}")

        # Tiempo de inferencia
        print(f"{'Tiempo inferencia (s)':<25} {results['fp16']['inference_time']:<15.2f} "
              f"{results['int8']['inference_time']:<15.2f} {results['int4']['inference_time']:<15.2f}")

        print(f"{'-'*80}")

        # Mejores en cada categoría
        fastest = min(results.items(), key=lambda x: x[1]['inference_time'])
        smallest = min(results.items(), key=lambda x: x[1]['size_mb'])

        print(f"\n🏆 Más rápido: {fastest[0].upper()}")
        print(f"🏆 Más pequeño: {smallest[0].upper()}")


def main():
    """Función principal de demostración"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║      DEMOSTRACIÓN DE CUANTIZACIÓN DE MODELOS DE LENGUAJE      ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # Usar un modelo pequeño para la demostración
    model_name = "microsoft/phi-2"  # 2.7B parámetros - bueno para demos

    quantizer = ModelQuantizer(model_name)

    test_prompt = "La inteligencia artificial es"

    # Ejecutar comparación completa
    results = quantizer.compare_all_quantizations(test_prompt)

    print("\n✅ Demostración completada")
    print("\n📝 CONCLUSIONES:")
    print("   • INT4 ofrece la mejor reducción de tamaño (~87.5%)")
    print("   • INT8 ofrece buen balance entre tamaño y calidad")
    print("   • FP16 mantiene mejor calidad pero usa más memoria")
    print("   • La cuantización permite ejecutar modelos grandes en hardware limitado")


if __name__ == "__main__":
    main()
