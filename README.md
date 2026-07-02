# Cómputo Híbrido CPU-GPU — CUDA y OpenMP Avanzado

**Actividad:** Semana 08 — Programación en GPU con CUDA y OpenMP Avanzado
**Curso:** Programación Paralela y Distribuida
**Institución:** Universidad Iberoamericana (UNIBE)
**Autor:** Pedribel Pion Rijo
**Matrícula:** 24-0429

---

## 1. Descripción

Programa híbrido en C++/CUDA que suma dos vectores de gran tamaño, `C[i] = A[i] + B[i]`, ejecutando la operación por dos rutas dentro del mismo binario:

| Ruta | Tecnología | Función |
|---|---|---|
| CPU | OpenMP | `add_vectors_cpu()` — `#pragma omp parallel for` |
| GPU | CUDA | `add_vectors()` — kernel `__global__` |

Al final se verifica que ambos resultados coincidan y se calcula el speedup de GPU sobre CPU.

---

## 2. Estructura del repositorio

```
cudaopenmp-hibrido/
├── README.md
├── Makefile
├── .gitignore
├── src/
│   └── suma_vectores_hibrido.cu
├── bin/                              (generado por make)
└── docs/
    └── ejemplo_salida.txt
```

---

## 3. Requisitos

| Herramienta | Versión mínima |
|---|---|
| CUDA Toolkit | 11.0+ |
| GPU NVIDIA | Compute Capability 6.0+ |
| g++ | 9.0+ |
| OpenMP | 4.5+ |
| GNU make | cualquiera |

Instalación en Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential
# CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
```

Verificación:

```bash
nvcc --version
nvidia-smi
```

---

## 4. Compilación

```bash
make                  # arquitectura por defecto (sm_75)
make ARCH=sm_86        # GPUs Ampere (RTX 30xx)
make run               # compila y ejecuta
make clean             # elimina binarios
make help              # lista de targets
```

Compilación manual equivalente:

```bash
nvcc -O3 -Xcompiler -fopenmp -std=c++17 -arch=sm_75 \
     src/suma_vectores_hibrido.cu -o bin/suma_vectores_hibrido
```

---

## 5. Ejecución

```bash
./bin/suma_vectores_hibrido
```

Parámetros fijos del programa: `N = 2^20 = 1,048,576` elementos `float`; configuración de kernel `4096 bloques × 256 hilos/bloque`.

---

## 6. Salida esperada

Ver [`docs/ejemplo_salida.txt`](docs/ejemplo_salida.txt). Resumen de campos impresos:

1. Configuración del problema (N, hilos CPU, bloques/hilos GPU).
2. Tiempo de la fase CPU (OpenMP).
3. Tiempo de la fase GPU (CUDA, incluye transferencias).
4. Verificación de correctitud (diferencia máxima entre resultados).
5. Speedup GPU/CPU.

> Los tiempos varían según hardware. Para N = 1,048,576 el overhead de transferencia y lanzamiento de kernel limita el speedup de la GPU; este comportamiento se documenta en la sección 6 del informe técnico que acompaña este repositorio.

---

## 7. Referencias de diseño (resumen)

| Elemento | Detalle |
|---|---|
| `schedule(static)` | Reparto uniforme entre hilos; costo por iteración constante |
| `if (idx < n)` | Guarda de límite en el kernel CUDA |
| `cudaMalloc` / `cudaMemcpy` / `cudaFree` | Ciclo de vida de memoria en el device |
| `reduction(&&:)` / `reduction(max:)` | Verificación paralela de resultados |

El desarrollo completo, con explicación línea por línea, está en el informe técnico `Informe_Pedribel_CUDA_OpenMP.docx`.

---

## 8. Documentos relacionados

- `Informe_Pedribel_CUDA_OpenMP.docx` — marco teórico, código explicado, resultados y speedup.
- `Presentacion_Pedribel_CUDA_OpenMP.pptx` — diapositivas y guion del video.

---

## 9. Autor

Pedribel Pion Rijo · 24-0429 · Universidad Iberoamericana (UNIBE) · Julio 2026
