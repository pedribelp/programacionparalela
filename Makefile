# =============================================================================
#  Makefile — Cómputo Híbrido CPU-GPU (OpenMP + CUDA)
#  Semana 08 · Programación Paralela y Distribuida · UNIBE
#  Autor: Pedribel Pion Rijo (24-0429)
# =============================================================================

NVCC      := nvcc
SRC_DIR   := src
BIN_DIR   := bin
TARGET    := $(BIN_DIR)/suma_vectores_hibrido
SRC       := $(SRC_DIR)/suma_vectores_hibrido.cu

# -Xcompiler -fopenmp reenvia la bandera de OpenMP al compilador de host (g++)
NVCC_FLAGS := -O3 -Xcompiler -fopenmp -std=c++17

# Arquitectura GPU objetivo (ajustar segun la tarjeta disponible)
# sm_75 Turing | sm_86 Ampere | sm_89 Ada Lovelace
ARCH ?= sm_75

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "make            - compila el programa hibrido"
	@echo "make run        - compila y ejecuta"
	@echo "make clean      - elimina los binarios"
	@echo "make ARCH=sm_86 - compila para una arquitectura GPU especifica"
