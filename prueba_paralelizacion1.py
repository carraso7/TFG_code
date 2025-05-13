# -*- coding: utf-8 -*-
"""
Created on Mon May 12 00:16:26 2025

@author: carlo
"""

import os
print("Estoy en:", os.getcwd())


import ctypes
import numpy as np

# Ruta absoluta (ajusta si es distinta)
lib_path = r"C:\Users\carlo\Documents\uni\TFG\Planar_graphs\TFG_code\libprueba_paralelizacion1.dll"
lib = ctypes.CDLL(lib_path)


# Define the argument types and return type
lib.square_matrix.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

# Create a sample matrix
n = 3
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], dtype=np.float64)
result = np.zeros((n, n), dtype=np.float64)

# Call the C function
lib.square_matrix(matrix, result, n)

print("Original matrix:")
print(matrix)

print("Squared matrix:")
print(result)
