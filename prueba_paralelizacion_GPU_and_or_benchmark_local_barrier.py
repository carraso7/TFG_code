# -*- coding: utf-8 -*-
"""
Created on Thu May 15 19:39:27 2025

@author: carlo
"""
import numpy as np
import time
import pyopencl as cl
import pandas as pd

# Configurar OpenCL
platforms = cl.get_platforms()
gpu_devices = [d for p in platforms for d in p.get_devices() if d.type & cl.device_type.GPU]
device = gpu_devices[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

# Kernel con reducción paralela usando __local + barrier
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

__kernel void parallel_reduce_or(__global const char* partial,
                                 __global char* C,
                                 const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int lid = get_local_id(0);
    int offset = row * N * N + col * N;

    __local char temp[1024];  // Asegúrate que N <= 1024

    if (lid < N)
        temp[lid] = partial[offset + lid];
    else
        temp[lid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = N / 2; s > 0; s >>= 1) {
        if (lid < s)
            temp[lid] |= temp[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        C[row * N + col] = temp[0];
}
"""

# Compilar kernel
program = cl.Program(ctx, KERNEL_CODE).build()
mf = cl.mem_flags

# Función de test
def test_gpu_vs_cpu(N):
    A = np.random.randint(0, 2, size=(N, N)).astype(np.int8)
    B = A.T.copy()
    C_gpu = np.zeros((N, N), dtype=np.int8)
    C_cpu = np.zeros((N, N), dtype=np.int8)

    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    partial_buf = cl.Buffer(ctx, mf.READ_WRITE, size=N*N*N)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=C_gpu.nbytes)

    # GPU
    start_gpu = time.time()
    program.partial_and_kernel(queue, (N, N, N), None, A_buf, B_buf, partial_buf, np.int32(N))
    program.parallel_reduce_or(queue, (N, N, N), (1, 1, N), partial_buf, C_buf, np.int32(N))
    # wg_limit = min(N, device.max_work_group_size)
    # program.parallel_reduce_or(queue, (N, N, wg_limit), (1, 1, wg_limit), ...)

    queue.finish()
    cl.enqueue_copy(queue, C_gpu, C_buf)
    gpu_time = time.time() - start_gpu

    # CPU
    start_cpu = time.time()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if A[i, k] & B[k, j]:
                    C_cpu[i, j] = 1
                    break
    cpu_time = time.time() - start_cpu

    return N, gpu_time, cpu_time

# Ejecutar para varios tamaños
for size in [64, 128, 256]:
    N, gpu_t, cpu_t = test_gpu_vs_cpu(size)
    print(f"Size: {N} | GPU: {gpu_t:.4f}s | CPU: {cpu_t:.4f}s")
