# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:39:27 2025

@author: carlo
"""




import pyopencl as cl
import numpy as np

# Inicialización
platforms = cl.get_platforms()
gpu_devices = [d for p in platforms for d in p.get_devices() if d.type & cl.device_type.GPU]
device = gpu_devices[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
print("Usando:", device.name)

# Tamaño de la matriz (ideal: potencia de 2 para futuras mejoras)
N = 4
A = np.random.randint(0, 2, size=(N, N)).astype(np.int8)
B = A.T.copy()
C = np.zeros((N, N), dtype=np.int8)

# Código OpenCL con reducción manual
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
        partial[idx] = A[row * N + k] & B[k * N + col];
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
        result = result | partial[offset + k];
        if (result) break;
    }
    C[row * N + col] = result;
}
"""

# Construir y preparar buffers
program = cl.Program(ctx, KERNEL_CODE).build()
mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
partial_buf = cl.Buffer(ctx, mf.READ_WRITE, size=N*N*N)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)

# Ejecutar kernels
program.partial_and_kernel(queue, (N, N, N), None, A_buf, B_buf, partial_buf, np.int32(N))
program.reduce_or_kernel(queue, (N, N), None, partial_buf, C_buf, np.int32(N))
cl.enqueue_copy(queue, C, C_buf)

# Mostrar resultados
print("Matriz A:")
print(A)
print("\nMatriz B:")
print(B)
print("\nResultado OR-AND paralelo:")
print(C)
