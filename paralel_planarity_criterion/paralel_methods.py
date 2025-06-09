import pyopencl as cl
import numpy as np

@staticmethod
def multiply_or_and(A, B, device=None):
    """
    Performs an AND OR matrix multiplication in paralel between matrices A and 
    B. and returns the result matrix 

    Parameters
    ----------
    A : np.matrix n*n
        First matrix to multiply.
    B : np.matrix n*n
        Second matrix to multiply.
    device : pyopencl device
        Device to perform the paralel multiplication. If it is None, any 
        available GPU device is chosen. The default is None.

    Returns
    -------
    C : np.matrix n*n
        Result matrix.

    """
    
    if device is None:
        platforms = cl.get_platforms()
        gpu_devices = [d for p in platforms for d in p.get_devices() if d.type & cl.device_type.GPU]
        device = gpu_devices[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    N = len(A) 
    A = np.array(A, dtype=np.int8) 
    B = np.array(B, dtype=np.int8)     
    C = np.zeros((N, N), dtype=np.int8) 
    
    KERNEL_CODE = f"""
    __kernel void partial_and_kernel(__global const char* A,
                                     __global const char* B,
                                     __global char* partial,
                                     const int N) {{
        int row = get_global_id(0);
        int col = get_global_id(1);
        int k = get_global_id(2);
        int idx = row * N * N + col * N + k;

        if (row < N && col < N && k < N) {{
            char a_val = A[row * N + k];
            char b_val = B[k * N + col];
            partial[idx] = a_val & b_val;
            // printf("AND hilo (%d, %d, %d): A=%d, B=%d => %d\\n",
            //       row, col, k, a_val, b_val, partial[idx]);
        }}
    }}

    __kernel void parallel_reduce_or(__global const char* partial,
                                     __global char* output,
                                     const int N) {{
        int row = get_global_id(0);
        int col = get_global_id(1);
        int lid = get_local_id(2);
        int group_size = get_local_size(2);
        int offset = row * N * N + col * N;

        __local char temp[{N}];
        

        if (lid < N)
            temp[lid] = partial[offset + lid];
        else
            temp[lid] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = group_size / 2; s > 0; s >>= 1) {{
            if (lid < s) {{
                temp[lid] = temp[lid] | temp[lid + s];
            }}
            barrier(CLK_LOCAL_MEM_FENCE);
        }}

        if (lid == 0)
            output[row * N + col] = temp[0];
    }}
    """
    
    # Compile and prepare buffers
    program = cl.Program(ctx, KERNEL_CODE).build()
    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    partial_buf = cl.Buffer(ctx, mf.READ_WRITE, size=N*N*N)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=C.nbytes)
    
    
    program.partial_and_kernel(
        queue, (N, N, N), None, A_buf, B_buf, partial_buf, np.int32(N)
        )
    queue.finish()
    
    program.parallel_reduce_or(
        queue, (N, N, N), (1, 1, N), partial_buf, C_buf, np.int32(N)
        )
    queue.finish()
    
    # Copy result back to C.
    cl.enqueue_copy(queue, C, C_buf)
    
    return C