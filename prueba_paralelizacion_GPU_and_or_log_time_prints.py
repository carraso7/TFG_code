# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:39:27 2025

@author: carlo
"""

import pyopencl as cl
import numpy as np

# Obtener contexto y dispositivo GPU
platforms = cl.get_platforms()
gpu_devices = [d for p in platforms for d in p.get_devices() if d.type & cl.device_type.GPU]
device = gpu_devices[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
print("Usando dispositivo:", device.name)

# Tamaño de la matriz (puedes aumentar)
N = 4
A = np.random.randint(0, 2, size=(N, N)).astype(np.int8)
B = A.T.copy()
C = np.zeros((N, N), dtype=np.int8)

# ================= KERNEL OpenCL con prints =================
KERNEL_CODE = """
__kernel void partial_and_kernel(__global const char* A,
                                 __global const char* B,
                                 __global char* partial,
                                 const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int k = get_global_id(2);
    int idx = row * N * N + col * N + k;

    if (row < N && col < N && k < N) {
        char a_val = A[row * N + k];
        char b_val = B[k * N + col];
        partial[idx] = a_val & b_val;
        printf("AND hilo (%d, %d, %d): A=%d, B=%d => %d\\n",
               row, col, k, a_val, b_val, partial[idx]);
    }
}

__kernel void reduce_or_kernel(__global char* partial,
                               __global char* C,
                               const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int offset = row * N * N + col * N;

    char result = 0;
    for (int k = 0; k < N; ++k) {
        char val = partial[offset + k];
        result = result | val;
        printf("OR hilo (%d, %d), k=%d: parcial=%d, acum=%d\\n",
               row, col, k, val, result);
        if (result) break;
    }
    C[row * N + col] = result;
}
"""

# Compilar y preparar buffers
program = cl.Program(ctx, KERNEL_CODE).build()
mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
partial_buf = cl.Buffer(ctx, mf.READ_WRITE, size=N*N*N)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)

# Ejecutar los kernels
print("\n--- Ejecutando partial_and_kernel ---")
program.partial_and_kernel(queue, (N, N, N), None, A_buf, B_buf, partial_buf, np.int32(N))
queue.finish()

print("\n--- Ejecutando reduce_or_kernel ---")
program.reduce_or_kernel(queue, (N, N), None, partial_buf, C_buf, np.int32(N))
queue.finish()

# Copiar resultado de vuelta a host
cl.enqueue_copy(queue, C, C_buf)

# Mostrar matrices
print("\nMatriz A:")
print(A)
print("\nMatriz B:")
print(B)
print("\nResultado C (OR-AND):")
print(C)
