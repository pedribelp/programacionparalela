// =============================================================================
//  suma_vectores_hibrido.cu
//  Programa híbrido CPU (OpenMP) + GPU (CUDA) para la suma de dos vectores
//  grandes: C[i] = A[i] + B[i]
//
//  Estructura en tres fases, tal como exige la actividad:
//    FASE 1 -> Suma en CPU paralelizada con OpenMP (#pragma omp parallel for)
//    FASE 2 -> Suma en GPU mediante un kernel CUDA (add_vectors)
//    FASE 3 -> Comparación de rendimiento (cálculo del speedup CPU vs GPU)
//
//  Autor:  Pedribel Pion Rijo  (24-0429)
//  Curso:  Programación Paralela y Distribuida — Universidad Iberoamericana (UNIBE)
//  Semana: 8 — Programación en GPU con CUDA y OpenMP Avanzado
//
//  Compilación:  nvcc -O3 -Xcompiler -fopenmp suma_vectores_hibrido.cu -o suma_vectores_hibrido
//  Ejecución:    ./suma_vectores_hibrido
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
//  Macro de verificación de errores CUDA. Envuelve cada llamada al runtime de
//  CUDA y aborta el programa con un mensaje claro si la llamada falla, en
//  lugar de dejar que el error se propague silenciosamente.
// -----------------------------------------------------------------------------
#define CUDA_CHECK(llamada)                                                        \
    do {                                                                           \
        cudaError_t err = (llamada);                                               \
        if (err != cudaSuccess) {                                                  \
            std::cerr << "Error CUDA en " << __FILE__ << ":" << __LINE__ << " -> "  \
                      << cudaGetErrorString(err) << std::endl;                     \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                          \
    } while (0)

// -----------------------------------------------------------------------------
//  Kernel CUDA: cada hilo GPU calcula UN ÚNICO elemento del vector resultado.
//  El índice global se obtiene combinando el índice de bloque, el tamaño de
//  bloque y el índice de hilo dentro del bloque; el "guard" (idx < n) evita
//  que los hilos sobrantes del último bloque escriban fuera de los límites
//  del arreglo cuando N no es múltiplo exacto de threadsPerBlock.
// -----------------------------------------------------------------------------
__global__ void add_vectors(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // -------------------------------------------------------------------------
    //  Parámetros del problema
    // -------------------------------------------------------------------------
    const int N = 1048576;              // 1 M de elementos (1 048 576 = 2^20)
    const size_t bytes = N * sizeof(float);
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "=========================================================\n";
    std::cout << "  Suma hibrida de vectores: OpenMP (CPU) + CUDA (GPU)\n";
    std::cout << "=========================================================\n";
    std::cout << "Tamano del vector (N):     " << N << "\n";
    std::cout << "Hilos OpenMP disponibles:  " << omp_get_max_threads() << "\n";
    std::cout << "Hilos por bloque CUDA:     " << threadsPerBlock << "\n";
    std::cout << "Bloques por grid CUDA:     " << blocksPerGrid << "\n\n";

    // -------------------------------------------------------------------------
    //  Reserva de memoria en el HOST (RAM de la CPU)
    // -------------------------------------------------------------------------
    float* A = new float[N];
    float* B = new float[N];
    float* C_cpu = new float[N];   // resultado calculado con OpenMP
    float* C_gpu = new float[N];   // resultado calculado con CUDA (copiado de vuelta)

    // Inicialización reproducible de los vectores de entrada
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i) * 0.5f;
        B[i] = static_cast<float>(N - i) * 0.25f;
    }

    // =========================================================================
    //  FASE 1 — VERSIÓN CPU PARALELA CON OpenMP
    // =========================================================================
    double inicio_cpu = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        C_cpu[i] = A[i] + B[i];
    }

    double tiempo_cpu = omp_get_wtime() - inicio_cpu;

    std::cout << "[FASE 1 - CPU / OpenMP]\n";
    std::cout << "  Tiempo de ejecucion: " << std::fixed << std::setprecision(6)
              << tiempo_cpu << " s\n\n";

    // =========================================================================
    //  FASE 2 — VERSIÓN GPU CON CUDA
    // =========================================================================
    float *d_A, *d_B, *d_C;

    // Eventos CUDA para medir con precisión de microsegundos en el dispositivo
    cudaEvent_t ev_inicio, ev_fin;
    CUDA_CHECK(cudaEventCreate(&ev_inicio));
    CUDA_CHECK(cudaEventCreate(&ev_fin));

    CUDA_CHECK(cudaEventRecord(ev_inicio));

    // Reserva de memoria en la GPU (device)
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Transferencia Host -> Device
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    // Lanzamiento del kernel: blocksPerGrid bloques de threadsPerBlock hilos
    add_vectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());       // errores de lanzamiento
    CUDA_CHECK(cudaDeviceSynchronize());  // errores de ejecucion + espera

    // Transferencia Device -> Host
    CUDA_CHECK(cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev_fin));
    CUDA_CHECK(cudaEventSynchronize(ev_fin));

    float tiempo_gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tiempo_gpu_ms, ev_inicio, ev_fin));
    double tiempo_gpu = tiempo_gpu_ms / 1000.0;  // total GPU: malloc + H2D + kernel + D2H

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(ev_inicio));
    CUDA_CHECK(cudaEventDestroy(ev_fin));

    // Verificación de correctitud: C_gpu[i] == A[i] + B[i]
    double max_diff = 0.0;
    bool correcto = true;
    for (int i = 0; i < N; i++) {
        double esperado = static_cast<double>(A[i]) + static_cast<double>(B[i]);
        double diff = std::fabs(static_cast<double>(C_gpu[i]) - esperado);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3) correcto = false;
    }

    std::cout << "[FASE 2 - GPU / CUDA]\n";
    std::cout << "  Tiempo total (malloc+H2D+kernel+D2H): "
              << std::fixed << std::setprecision(6) << tiempo_gpu << " s\n";
    std::cout << "  Diferencia maxima vs. valor esperado: "
              << std::scientific << max_diff << "\n";
    std::cout << std::fixed;
    std::cout << "  Verificacion:        " << (correcto ? "CORRECTO" : "INCORRECTO") << "\n\n";

    // =========================================================================
    //  FASE 3 — COMPARACIÓN DE RENDIMIENTO
    // =========================================================================
    double speedup = tiempo_cpu / tiempo_gpu;

    std::cout << "[FASE 3 - RENDIMIENTO]\n";
    std::cout << std::setprecision(2);
    std::cout << "  Tiempo CPU (OpenMP): " << std::setprecision(6) << tiempo_cpu << " s\n";
    std::cout << "  Tiempo GPU (CUDA):   " << tiempo_gpu << " s\n";
    std::cout << std::setprecision(2);
    std::cout << "  Speedup (CPU/GPU):   " << speedup << "x\n";
    std::cout << "=========================================================\n";

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;

    return 0;
}
