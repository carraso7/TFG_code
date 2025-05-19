# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:39:27 2025

@author: carlo
"""

import pyopencl as cl
import numpy as np

# Define el código OpenCL (kernel para multiplicación de matrices)
KERNEL_CODE = """
__kernel void mat_square(__global const double* A,
                         __global double* C,
                         const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * A[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

# =================== CONFIGURACIÓN DE DISPOSITIVO ===================
platforms = cl.get_platforms()
gpu_devices = [d for p in platforms for d in p.get_devices()
               if d.type & cl.device_type.GPU]

if not gpu_devices:
    raise RuntimeError("No se encontró ninguna GPU compatible con OpenCL.")

device = gpu_devices[0]

print("Usando dispositivo:", device.name)

print()
print("Especifiaciones:")
print("Nombre:", device.name)
print("Tipo:", cl.device_type.to_string(device.type))
print("Memoria global (MB):", device.global_mem_size // 1024**2)
print("Unidades de cómputo:", device.max_compute_units)
print()

ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

# =================== DATOS ===================
N = 3
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=np.float64)
C = np.zeros((N, N), dtype=np.float64)

# =================== BUFFERS ===================
mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

# =================== COMPILACIÓN Y EJECUCIÓN ===================
program = cl.Program(ctx, KERNEL_CODE).build()
program.mat_square(queue, (N, N), None, A_buf, C_buf, np.int32(N))
cl.enqueue_copy(queue, C, C_buf)

# =================== RESULTADO ===================
print("Matriz original:")
print(A)
print("\nMatriz al cuadrado (GPU OpenCL):")
print(C)
