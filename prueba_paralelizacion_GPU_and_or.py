# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:39:27 2025

@author: carlo
"""
import pyopencl as cl
import numpy as np

KERNEL_CODE1 = """
__kernel void mat_or_and(__global const char* A,
                         __global const char* B,
                         __global char* C,
                         const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < N && col < N) {
        char result = 0;
        for (int k = 0; k < N; ++k) {
            result = result || (A[row * N + k] & B[k * N + col]);
            if (result) break;  // short-circuit if already 1
        }
        C[row * N + col] = result;
    }
}
"""

# Código con prints
KERNEL_CODE = """
__kernel void mat_or_and(__global const char* A,
                         __global const char* B,
                         __global char* C,
                         const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    printf("Hilo activo: row = %d, col = %d\\n", row, col);

    if (row < N && col < N) {
        char result = 0;
        for (int k = 0; k < N; ++k) {
            result = result || (A[row * N + k] & B[k * N + col]);
            if (result) break;
        }
        C[row * N + col] = result;
    }
}
"""


# Configurar plataforma y GPU
platforms = cl.get_platforms()
for platform in platforms:
    print(f"Plataforma: {platform.name}")
    for device in platform.get_devices():
        print(f"  Dispositivo: {device.name}")
        print(f"    Tipo: {cl.device_type.to_string(device.type)}")
        print(f"    Unidades de cómputo: {device.max_compute_units}")
        print(f"    Tamaño de grupo de trabajo: {device.max_work_group_size}")


gpu_devices = [d for p in platforms for d in p.get_devices() if d.type & cl.device_type.GPU]
if not gpu_devices:
    raise RuntimeError("No GPU compatible con OpenCL encontrada.")
device = gpu_devices[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

print("Usando dispositivo:", device.name)

print()
print("Especifiaciones:")
print("Nombre:", device.name)
print("Tipo:", cl.device_type.to_string(device.type))
print("Memoria global (MB):", device.global_mem_size // 1024**2)
print("Unidades de cómputo:", device.max_compute_units)
print("Máx. hilos por grupo (work-group):", device.max_work_group_size)
print()

# Ejemplo de matrices binarias
N = 4
A = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 0, 0],
              [0, 0, 1, 1]], dtype=np.int8)
B = A.T.copy()  # puedes cambiar esto
C = np.zeros((N, N), dtype=np.int8)

# Buffers
mf = cl.mem_flags
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)

# Compilar y ejecutar
program = cl.Program(ctx, KERNEL_CODE).build()
program.mat_or_and(queue, (N, N), None, A_buf, B_buf, C_buf, np.int32(N))
cl.enqueue_copy(queue, C, C_buf)

# Mostrar resultado
print("A:")
print(A)
print("\nB:")
print(B)
print("\nA (OR-AND) B:")
print(C)
